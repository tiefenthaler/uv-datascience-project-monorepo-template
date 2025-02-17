# An example using multi-stage image builds to create a final image without uv.
# -----------------------------------------------------------------------------
# Defines an optimized image to reduce image size.
# Custom python code should be a packaged application.
# ----------------------------------------------------

# Define a build-time argument with a default value for base container images.
ARG UV_VER=python3.12-bookworm-slim
# Define a build-time argument with a default value for the workspace name.
ARG WORKSPACE_NAME=app

# FIRST, build the application in the `/WORKSPACE_NAME` directory. See `Dockerfile` for details.
FROM ghcr.io/astral-sh/uv:$UV_VER AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /${WORKSPACE_NAME}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --no-dev
# Then, add the rest of the project source code and install it. See `Dockerfile` for details.
COPY . /${WORKSPACE_NAME}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages --no-dev

# SECOND, use a final image without uv.
FROM python:3.12-slim-bookworm
# It is important to use the image that matches the builder, as the path to the
# python executable must be the same, e.g., using `python:3.11-slim-bookworm` will fail.

# Copy the application from the builder
COPY --from=builder --chown=${WORKSPACE_NAME}:${WORKSPACE_NAME} /${WORKSPACE_NAME} /${WORKSPACE_NAME}

# Place executables in the environment at the front of the path
ENV PATH="/${WORKSPACE_NAME}/.venv/bin:$PATH"

# Run the FastAPI application by default with `uvicorn`.
CMD ["uv", "run", "hello"]
