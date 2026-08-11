#!/usr/bin/env python3
"""Serve Qwen + Cosmos3-Nano + pi0.5 as one openpi websocket policy."""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from cosmos_pi05.hierarchical_policy import HierarchicalPi05Policy
from cosmos_pi05.hierarchical_policy import build_orchestrator
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    config_name: str = "pi05_libero_long_subgoal"
    high_level_endpoint: str = "http://127.0.0.1:10090"
    cosmos_endpoint: str = "http://127.0.0.1:10091"
    port: int = 8000
    probability_threshold: float = 0.65
    memory_margin_threshold: float = 1.0
    branching_factor: int = 3
    beam_width: int = 2
    depth: int = 2
    world_profile: str = "deploy"
    denoising_steps: int = 10
    decision_timeout_s: float = 1800.0
    pytorch_device: str | None = None


def main(args: Args) -> None:
    if not args.cosmos_endpoint.strip():
        raise ValueError("cosmos_endpoint is required: this policy never falls back to a text-only visual goal")
    config = training_config.get_config(args.config_name)
    low_level = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": args.denoising_steps},
        pytorch_device=args.pytorch_device,
    )
    orchestrator = build_orchestrator(
        high_level_endpoint=args.high_level_endpoint,
        cosmos_endpoint=args.cosmos_endpoint,
        probability_threshold=args.probability_threshold,
        memory_margin_threshold=args.memory_margin_threshold,
        branching_factor=args.branching_factor,
        beam_width=args.beam_width,
        depth=args.depth,
        world_profile=args.world_profile,
        timeout_s=args.decision_timeout_s,
    )
    policy = HierarchicalPi05Policy(
        low_level,
        orchestrator,
        decision_timeout_s=args.decision_timeout_s,
        metadata={
            **(low_level.metadata or {}),
            "config_name": args.config_name,
            "checkpoint_dir": args.checkpoint_dir,
            "denoising_steps": args.denoising_steps,
        },
    )
    hostname = socket.gethostname()
    logging.info("Serving hierarchical pi0.5 on %s:%d", hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
