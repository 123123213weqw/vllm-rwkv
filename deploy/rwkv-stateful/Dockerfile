# syntax=docker/dockerfile:1.7
# Build/publish the CUDA-matched rwkv-vllm-native base separately. This layer
# contains only the stateful router/plugin sources, so deployment iteration does
# not rebuild CUDA extensions.
ARG RWKV_VLLM_BASE_IMAGE=ghcr.io/123123213weqw/rwkv-vllm-native:cu128
FROM ${RWKV_VLLM_BASE_IMAGE}

WORKDIR /opt/rwkv-vllm-native
COPY vllm/rwkv_stateful ./vllm/rwkv_stateful
COPY plugins/vllm-rwkv7 /tmp/vllm-rwkv7
RUN uv pip install --system --no-deps /tmp/vllm-rwkv7 && rm -rf /tmp/vllm-rwkv7

ENV RWKV_SESSION_REGISTRY=/var/lib/rwkv-router/sessions.db
EXPOSE 8000 8080
CMD ["python", "-m", "vllm.rwkv_stateful.router", "--port", "8080"]
