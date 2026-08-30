# syntax=docker/dockerfile:1.7
# Build/publish the CUDA-matched vllm-rwkv base separately. This layer
# contains only the stateful router/plugin sources, so deployment iteration does
# not rebuild CUDA extensions.
ARG RWKV_VLLM_BASE_IMAGE=ghcr.io/123123213weqw/vllm-rwkv:cu128
FROM ${RWKV_VLLM_BASE_IMAGE}

WORKDIR /opt/vllm-rwkv
COPY vllm/rwkv_stateful ./vllm/rwkv_stateful
COPY plugins/vllm-rwkv /tmp/vllm-rwkv
RUN uv pip install --system --no-deps /tmp/vllm-rwkv && rm -rf /tmp/vllm-rwkv

ENV RWKV_SESSION_REGISTRY=/var/lib/rwkv-router/sessions.db
EXPOSE 8000 8080
CMD ["python", "-m", "vllm.rwkv_stateful.router", "--port", "8080"]
