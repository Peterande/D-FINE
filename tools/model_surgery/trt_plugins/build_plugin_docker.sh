#!/usr/bin/env bash
set -euo pipefail

# Build GatherElements TRT plugin in a clean CUDA container to avoid
# host toolchain/glibc conflicts.
#
# Usage:
#   bash tools/model_surgery/trt_plugins/build_plugin_docker.sh \
#     /home/berna/tensorrt \
#     outputs/phase2_run/plugin_build_docker
#
# Args:
#   $1: TENSORRT_ROOT on host (must contain include/ and lib/)
#   $2: Build output dir (default: outputs/phase2_run/plugin_build_docker)

TRT_ROOT_HOST="${1:-/home/berna/tensorrt}"
OUT_DIR="${2:-outputs/phase2_run/plugin_build_docker}"

if [[ ! -d "${TRT_ROOT_HOST}/include" || ! -d "${TRT_ROOT_HOST}/lib" ]]; then
  echo "[error] Invalid TensorRT root: ${TRT_ROOT_HOST}"
  echo "        Expected: ${TRT_ROOT_HOST}/include and ${TRT_ROOT_HOST}/lib"
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_ABS="${REPO_ROOT}/${OUT_DIR}"
mkdir -p "${OUT_ABS}"

echo "[info] repo: ${REPO_ROOT}"
echo "[info] TRT root: ${TRT_ROOT_HOST}"
echo "[info] output: ${OUT_ABS}"

docker run --rm \
  -v "${REPO_ROOT}:/workspace/repo" \
  -v "${TRT_ROOT_HOST}:/workspace/tensorrt:ro" \
  -w /workspace/repo \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash -lc "
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends build-essential cmake
    cmake -S tools/model_surgery/trt_plugins \
          -B ${OUT_DIR} \
          -DTENSORRT_ROOT=/workspace/tensorrt
    cmake --build ${OUT_DIR} -j
    ls -lh ${OUT_DIR}
  "

echo
echo "[done] Built plugin artifacts in: ${OUT_ABS}"
echo "[done] Expected plugin .so:"
echo "       ${OUT_ABS}/libgather_elements_axis1_plugin.so"
