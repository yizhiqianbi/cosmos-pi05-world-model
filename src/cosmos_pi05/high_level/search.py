"""World-model-guided beam search over open-ended language subtasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import inf

from .interfaces import ProposalModel
from .interfaces import ValueModel
from .interfaces import WorldModel
from .types import HighLevelObservation
from .types import SearchBranch
from .types import SearchResult
from .types import ensure_unique_branch_ids


@dataclass(frozen=True)
class SearchConfig:
    branching_factor: int = 3
    beam_width: int = 2
    depth: int = 2
    world_profile: str = "deploy"

    def __post_init__(self) -> None:
        for name in ("branching_factor", "beam_width", "depth"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def deterministic_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFFFFFF


class WorldModelBeamSearch:
    """Faithful propose → predict → evaluate → global Top-B expansion."""

    def __init__(self, proposal: ProposalModel, world: WorldModel, value: ValueModel, config: SearchConfig):
        self.proposal = proposal
        self.world = world
        self.value = value
        self.config = config

    def run(self, root_context: HighLevelObservation) -> SearchResult:
        root = SearchBranch(
            branch_id="root",
            parent_id=None,
            depth=0,
            sample_index=0,
            path=(),
            memory=root_context.memory,
            local_score=0.0,
            cumulative_score=0.0,
            outcome=None,
        )
        beam = [root]
        expanded_nodes = 0
        invalid_nodes = 0

        for depth in range(1, self.config.depth + 1):
            contexts = [self._context_for_branch(root_context, branch) for branch in beam]
            proposal_seeds = [
                [
                    deterministic_seed(root_context.episode_id, root_context.sequence_id, depth, branch.branch_id, idx)
                    for idx in range(self.config.branching_factor)
                ]
                for branch in beam
            ]
            proposal_groups = self.proposal.generate_many(
                contexts,
                samples_per_context=self.config.branching_factor,
                seeds=proposal_seeds,
            )
            if len(proposal_groups) != len(beam):
                raise ValueError("Proposal model returned a different number of context groups")

            flat: list[tuple[SearchBranch, HighLevelObservation, object, int]] = []
            for parent, context, proposals in zip(beam, contexts, proposal_groups, strict=True):
                if len(proposals) != self.config.branching_factor:
                    raise ValueError("Proposal model did not return branching_factor samples")
                for sample_index, proposal in enumerate(proposals):
                    seed = deterministic_seed(
                        root_context.episode_id,
                        root_context.sequence_id,
                        "world",
                        depth,
                        parent.branch_id,
                        sample_index,
                    )
                    flat.append((parent, context, proposal, seed))

            outcomes = self.world.predict_many(
                [(context, proposal) for _, context, proposal, _ in flat],
                seeds=[seed for *_, seed in flat],
                profile=self.config.world_profile,
            )
            if len(outcomes) != len(flat):
                raise ValueError("World model returned a different number of outcomes")

            valid_value_requests = []
            valid_indices = []
            for idx, ((_, _, proposal, _), outcome) in enumerate(zip(flat, outcomes, strict=True)):
                if outcome.valid and outcome.terminal_image is not None:
                    valid_indices.append(idx)
                    valid_value_requests.append((root_context.task_instruction, proposal, outcome))
                else:
                    invalid_nodes += 1
            values = self.value.score_many(valid_value_requests) if valid_value_requests else []
            if len(values) != len(valid_value_requests):
                raise ValueError("Value model returned a different number of scores")
            values_by_index = dict(zip(valid_indices, values, strict=True))

            children: list[SearchBranch] = []
            for idx, ((parent, _, proposal, _), outcome) in enumerate(zip(flat, outcomes, strict=True)):
                value = values_by_index.get(idx)
                local_score = float(value.score) if value is not None else -inf
                branch_id = f"d{depth}:{parent.branch_id}:{proposal.sample_index}:{idx}"
                children.append(
                    SearchBranch(
                        branch_id=branch_id,
                        parent_id=parent.branch_id,
                        depth=depth,
                        sample_index=proposal.sample_index,
                        path=(*parent.path, proposal.subtask),
                        memory=proposal.memory,
                        local_score=local_score,
                        cumulative_score=parent.cumulative_score + local_score,
                        outcome=outcome,
                        proposal=proposal,
                        immediate_outcome=(outcome if parent.depth == 0 else parent.immediate_outcome),
                    )
                )
            expanded_nodes += len(children)
            ensure_unique_branch_ids(children)
            ranked = sorted(children, key=lambda branch: (-branch.cumulative_score, branch.branch_id))
            if depth == self.config.depth:
                beam = ranked[: self.config.beam_width]
            else:
                expandable = [
                    branch
                    for branch in ranked
                    if branch.outcome is not None and branch.outcome.valid and branch.outcome.terminal_image is not None
                ]
                if not expandable:
                    # Preserve the failed frontier for diagnostics, but never
                    # try to use an invalid terminal image as the next state.
                    beam = ranked[: self.config.beam_width]
                    break
                beam = expandable[: self.config.beam_width]

        return SearchResult(tuple(beam), expanded_nodes=expanded_nodes, invalid_nodes=invalid_nodes)

    @staticmethod
    def _context_for_branch(root_context: HighLevelObservation, branch: SearchBranch) -> HighLevelObservation:
        if branch.depth == 0:
            return root_context
        if branch.outcome is None or not branch.outcome.valid or branch.outcome.terminal_image is None:
            raise ValueError(f"Cannot expand invalid branch {branch.branch_id}")
        return root_context.with_branch_state(
            memory=branch.memory,
            previous_subtask=branch.path[-1],
            head_image=branch.outcome.terminal_image,
        )
