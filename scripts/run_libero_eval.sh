#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_VENV="${LIBERO_EVAL_VENV:-${REPOSITORY_ROOT}/.venv-libero}"

if [[ ! -x "${EVAL_VENV}/bin/python" ]]; then
  echo "Missing ${EVAL_VENV}; run scripts/setup_libero_eval_env.sh first." >&2
  exit 2
fi

export PYTHONPATH="${REPOSITORY_ROOT}/packages/openpi-client/src:${REPOSITORY_ROOT}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
cd "${REPOSITORY_ROOT}"
exec "${EVAL_VENV}/bin/python" examples/libero/eval_hierarchical.py "$@"
