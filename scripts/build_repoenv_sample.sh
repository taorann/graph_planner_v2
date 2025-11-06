#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  echo "This script must be run with bash (try 'bash $0')." >&2
  exit 1
fi

set -euo pipefail

# Build the local RepoEnv-compatible sample image used by the rule-based agent
# smoke tests. The resulting image tag is referenced from
# datasets/graphplanner_repoenv_sample.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTEXT_DIR="${REPO_ROOT}/docker/repoenv_sample"
IMAGE_TAG="graph-planner/repoenv-sample:latest"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build the RepoEnv sample image" >&2
  exit 1
fi

# Ensure the git metadata copied into the container is in a clean state.
rm -f "${CONTEXT_DIR}/sample_repo/.pytest_cache" >/dev/null 2>&1 || true
rm -rf "${CONTEXT_DIR}/sample_repo/__pycache__" >/dev/null 2>&1 || true

DOCKER_BUILDKIT=1 docker build \
  --tag "${IMAGE_TAG}" \
  "${CONTEXT_DIR}"

echo "Built ${IMAGE_TAG}. Update your RepoEnv dataset descriptor to use this tag."
