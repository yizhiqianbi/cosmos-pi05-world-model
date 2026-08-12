#!/usr/bin/env bash
# Backward-compatible name for the standalone visual-subgoal trainer.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPOSITORY_ROOT}/scripts/train_pi05_subgoal.sh" "$@"
