# Repository Guidelines

## Project Structure & Module Organization

Core Python packages live under `src/`. Upstream policy, model, training, and serving code is in `src/openpi/`; Cosmos integration, hierarchical policy logic, and differentiable co-training are in `src/cosmos_pi05/`. Executable workflows belong in `scripts/`, with specialized helpers in `scripts/cosmos/` and `scripts/cotrain/`. Environment-specific clients and data converters are under `examples/`; the reusable client package is in `packages/openpi-client/`. Architecture and operational notes live in `docs/`. Treat `third_party/` as vendored code and avoid incidental edits. Local datasets, checkpoints, outputs, and virtual environments are intentionally not versioned.

## Build, Test, and Development Commands

- `GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --dev` installs Python 3.11 dependencies, workspace packages, and development tools.
- `uv run pytest --strict-markers -m "not manual"` runs the standard CI test suite.
- `uv run pytest -q src/cosmos_pi05 scripts/cotrain` runs the focused Cosmos/co-training tests documented in the README.
- `uv run ruff check .` checks imports, naming, and lint rules; `uv run ruff format .` formats Python sources.
- `pre-commit run --all-files` reproduces repository-wide pre-commit checks.

Use the launch scripts documented in `README.md` for GPU workflows, for example `scripts/cotrain_cosmos_pi05_e2e_torch.sh --max-steps 1 --save-interval 1` for a minimal joint-training smoke test. These jobs require dedicated environments and substantial GPU memory.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.11 syntax, and a 120-character line limit. Ruff enforces formatting and linting; do not hand-format around it. Follow `snake_case` for modules, functions, and tests, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep environment-specific behavior in policies or scripts rather than adding conditionals to shared model code. Preserve typed-array annotations and existing type hints.

## Testing Guidelines

Pytest discovers tests in `src`, `scripts`, and `packages`. Place tests beside implementation files and name them `*_test.py`; name test functions `test_<behavior>`. Mark hardware-, data-, or operator-driven tests with `@pytest.mark.manual`. Add focused unit tests for changes to transforms, policy configuration, datasets, or co-training gradients. No numeric coverage threshold is configured, but affected paths should be exercised.

## Commit & Pull Request Guidelines

Recent history favors concise, imperative subjects, often with Conventional Commit prefixes such as `feat:` or `docs:`. Keep commits scoped and avoid checking in checkpoints, generated artifacts, or datasets. Pull requests should explain motivation and behavior, link relevant issues, list validation commands, and call out hardware or dataset assumptions. Include logs or screenshots when changing training results, services, or user-visible evaluation behavior. Ensure tests, Ruff, and pre-commit pass before requesting review.
