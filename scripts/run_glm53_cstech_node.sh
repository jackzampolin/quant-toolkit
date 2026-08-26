#!/usr/bin/env bash
set -euo pipefail

# Launch one node of the pinned two-host GLM-5.3-Flash cstech runtime.
# Start node rank 1 first, then node rank 0. The pinned sparse-SM120 backend
# currently supports only the effective fp8_ds_mla cache layout.

IMAGE="${IMAGE:-cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5}"
MODEL_DIR="${MODEL_DIR:-/data/models/GLM-5.3-Flash-BF16-b1967181}"
MODEL_REVISION="${MODEL_REVISION:-b1967181a3917ae70a437f4884748f6b8e3a1f4d}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-BF16}"

NODE_RANK="${NODE_RANK:?set NODE_RANK to 0 (head) or 1 (worker)}"
NNODES="${NNODES:-2}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-10.42.20.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
HEAD_HOST_IP="${HEAD_HOST_IP:-10.42.20.1}"
WORKER_HOST_IP="${WORKER_HOST_IP:-10.42.20.2}"
NCCL_IFACE="${NCCL_IFACE:-enp33s0f1np1}"

PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MTP_TOKENS="${MTP_TOKENS:-5}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-7200}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
CPU_DISTRIBUTED_TIMEOUT_SECONDS="${CPU_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
CAPTURE_DIR="${CAPTURE_DIR:-}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-}"

case "${NODE_RANK}" in
  0)
    ROLE="head"
    HOST_IP="${HEAD_HOST_IP}"
    ;;
  1)
    ROLE="worker"
    HOST_IP="${WORKER_HOST_IP}"
    ;;
  *)
    echo "NODE_RANK must be 0 or 1, got ${NODE_RANK}" >&2
    exit 2
    ;;
esac

if [[ "${NNODES}" != 2 || "${TP_SIZE}" != 8 ]]; then
  echo "This sealed campaign launcher requires NNODES=2 and TP_SIZE=8." >&2
  exit 2
fi
if ((DCP_SIZE < 1 || TP_SIZE % DCP_SIZE != 0)); then
  echo "DCP_SIZE must be a positive divisor of TP_SIZE." >&2
  exit 2
fi
if [[ "${KV_CACHE_DTYPE}" != fp8 ]]; then
  echo "The pinned sparse-SM120 backend only supports effective fp8_ds_mla; requested ${KV_CACHE_DTYPE}." >&2
  exit 2
fi
if [[ ! "${PREFIX_CACHING}" =~ ^[01]$ ]]; then
  echo "PREFIX_CACHING must be 0 or 1." >&2
  exit 2
fi
if ((MTP_TOKENS < 0)); then
  echo "MTP_TOKENS must be non-negative." >&2
  exit 2
fi
if [[ -n "${CAPTURE_DIR}" ]]; then
  if [[ "${PREFIX_CACHING}" != 0 || "${MTP_TOKENS}" != 0 || "${MAX_NUM_SEQS}" != 1 ]]; then
    echo "Capture requires PREFIX_CACHING=0, MTP_TOKENS=0, and MAX_NUM_SEQS=1." >&2
    exit 2
  fi
  mkdir -p "${CAPTURE_DIR}"
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing model directory: ${MODEL_DIR}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "Missing model index under ${MODEL_DIR}." >&2
  exit 2
fi

CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-bf16-tp8-${ROLE}}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache/vllm-glm53-flash-bf16-tp8}"
mkdir -p "${CACHE_DIR}"

ACTUAL_IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
if [[ -n "${EXPECTED_IMAGE_ID}" && "${ACTUAL_IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Image ID mismatch: expected ${EXPECTED_IMAGE_ID}, got ${ACTUAL_IMAGE_ID}." >&2
  exit 2
fi
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

capture_args=()
if [[ -n "${CAPTURE_DIR}" ]]; then
  capture_args=(
    -e VLLM_KLD_HIDDEN_CAPTURE_DIR=/capture-hidden
    -v "${CAPTURE_DIR}:/capture-hidden:rw"
  )
fi
prefix_args=()
if [[ "${PREFIX_CACHING}" == 1 ]]; then
  prefix_args=(--enable-prefix-caching)
fi
speculative_args=()
if ((MTP_TOKENS > 0)); then
  speculative_args=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
  )
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --init \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VLLM_ENGINE_READY_TIMEOUT_S="${READY_TIMEOUT_SECONDS}" \
  -e OMP_NUM_THREADS=2 \
  -e VLLM_HOST_IP="${HOST_IP}" \
  -e GLOO_SOCKET_IFNAME="${NCCL_IFACE}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_IFACE}" \
  -e NCCL_NET=Socket \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_ENV_PLUGIN=none \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_RMA_PLUGIN=none \
  -e NCCL_GIN_PLUGIN=none \
  "${capture_args[@]}" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${CACHE_DIR}:/root/.cache" \
  "${IMAGE}" \
  /model \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --distributed-executor-backend mp \
  --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS}" \
  --cpu-distributed-timeout-seconds "${CPU_DISTRIBUTED_TIMEOUT_SECONDS}" \
  --disable-custom-all-reduce \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  "${prefix_args[@]}" \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"reasoning_effort":"max"}' \
  "${speculative_args[@]}"

cat <<EOF
Started ${CONTAINER_NAME} (node rank ${NODE_RANK})
Image: ${IMAGE}
Model revision: ${MODEL_REVISION}
TP=${TP_SIZE} DCP=${DCP_SIZE} effective KV=fp8_ds_mla
Image ID: ${ACTUAL_IMAGE_ID}
EOF
