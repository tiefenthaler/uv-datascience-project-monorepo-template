# Docker Production Setup

This document provides an in-depth guide to the Docker production setup used in this project. It covers the rationale behind the chosen approach, the structure of the `Dockerfile` respectively `multistage.Dockerfile` and `docker-compose.yml` files, and best practices for deploying the application in a production environment.

- [Docker Production Setup](#docker-production-setup)
  - [Overview](#overview)
    - [Key Components](#key-components)
  - [`Dockerfile` Explained](#dockerfile-explained)
    - [1. Base Image](#1-base-image)
    - [2. Working Directory](#2-working-directory)
    - [3. Dependency Installation](#3-dependency-installation)
    - [4. Copying Application Code](#4-copying-application-code)
    - [5. Installing Application](#5-installing-application)
    - [6. Environment Variables](#6-environment-variables)
    - [7. Entrypoint and Command](#7-entrypoint-and-command)
  - [`multistage.Dockerfile` Explained](#multistagedockerfile-explained)
    - [1. Builder Stage Image](#1-builder-stage-image)
    - [2. Final Image](#2-final-image)
  - [`docker-compose.yml` Explained](#docker-composeyml-explained)
    - [Build Arguments](#build-arguments)
    - [1. Standard Services](#1-standard-services)
    - [2. Optimized Docker Service](#2-optimized-docker-service)
  - [Build and Run and Test the Docker Application](#build-and-run-and-test-the-docker-application)
  - [Best Practices](#best-practices)
  - [Getting Started](#getting-started)
  - [Conclusion](#conclusion)

## Overview

The Docker setup is designed to create a lean, efficient, and secure production environment for the application. It leverages multi-stage builds to minimize the final image size and uses `uv` for fast and reliable dependency management.

### Key Components

1. **`Dockerfile`:** Defines the steps to build the Docker image. It includes:

   - Base image selection.
   - Dependency installation using `uv`.
   - Copying application code.
   - Setting environment variables.
   - Defining the entry point and command.

2. **`multistage.Dockerfile`:** Defines an optimized image to reduce image size. Custom python code should be a packaged application.

3. **`docker-compose.yml`:** Defines the services, networks, and volumes for the application. It includes:

   - Service definitions for the application.
   - Port mappings.
   - Environment variables.
   - Build configurations.

## `Dockerfile` Explained

The standard `Dockerfile` is designed to create a production-ready image that includes all necessary dependencies and the application code. It leverages the uv package manager for efficient dependency management and environment setup. The `Dockerfile` is structured to optimize for layer caching and minimize the final image size. Here's a breakdown of the key sections:

### 1. Base Image

We use a `uv` pre-installed Python image with a Debian base. The `UV_VER` argument allows us to specify the Python version at build time.

```dockerfile
FROM ghcr.io/astral-sh/uv:$UV_VER AS uv
```

### 2. Working Directory

Sets the working directory inside the container.

```dockerfile
WORKDIR /${WORKSPACE_NAME}
```

### 3. Dependency Installation

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --no-dev
```

Installs the project dependencies using `uv sync`:

- `--frozen` ensures that the exact versions specified in `uv.lock` are used.
- `--no-install-project` prevents the project itself from being installed at this stage.
- `--no-dev` excludes development dependencies.
- `--mount=type=cache,target=/root/.cache/uv` caches the uv environment.
- `--mount=type=bind,source=uv.lock,target=uv.lock` binds the uv.lock file to the container.
- `--mount=type=bind,source=pyproject.toml,target=pyproject.toml` binds the pyproject.toml file to the container.

### 4. Copying Application Code

Copies the application code into the container.

```dockerfile
COPY . /${WORKSPACE_NAME}
```

### 5. Installing Application

Installs the project itself. `--mount=type=cache,target=/root/.cache/uv` caches the uv environment.

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-packages --frozen --no-dev
```

### 6. Environment Variables

Adds the virtual environment's `bin` directory to the `PATH` environment variable, ensuring that the correct executables are used.

```dockerfile
ENV PATH="/${WORKSPACE_NAME}/.venv/bin:$PATH"
```

### 7. Entrypoint and Command

Sets the default command to run when the container starts, which in this case is starting the FastAPI application using `uvicorn`.

```dockerfile
ENTRYPOINT []
CMD ["uv", "run", "hello"]
```

## `multistage.Dockerfile` Explained

The `multistage.Dockerfile` is designed to create an optimized production image by separating the build and runtime environments. This approach reduces the final image size and ensures that only the necessary components are included in the runtime image. Here's a breakdown of the key sections:

### 1. Builder Stage Image

Uses the `uv` image to build the application and install dependencies including the packaged project application.

```dockerfile
FROM ghcr.io/astral-sh/uv:$UV_VER AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /${WORKSPACE_NAME}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --no-dev
COPY . /${WORKSPACE_NAME}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages --no-dev
```

### 2. Final Image

Uses a slim Python image for the runtime environment and copies the built application from the builder stage.

```dockerfile
FROM python:3.12-slim-bookworm
COPY --from=builder --chown=${WORKSPACE_NAME}:${WORKSPACE_NAME} /${WORKSPACE_NAME} /${WORKSPACE_NAME}
ENV PATH="/${WORKSPACE_NAME}/.venv/bin:$PATH"
CMD ["uv", "run", "hello"]
```

## `docker-compose.yml` Explained

The `docker-compose.yml` file defines the services, networks, and volumes for the application. Here's a breakdown of the key sections:

### Build Arguments

Defines reusable build arguments for the Dockerfile.

```yaml
x-args: &default-args
  WORKSPACE_NAME: "app"
  UV_VER: "python3.12-bookworm"
```

### 1. Standard Services

Builds and runs the application using the standard Dockerfile.

- Defines the `app` service, which is built from the `Dockerfile` in the current directory.
- The `args` section passes build arguments to the `Dockerfile`.
- The `ports` section maps port 8000 on the host to port 8000 on the container.
- The `command` section specifies the command to run when the container starts.

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        <<: *default-args
    image: lit_autoencoder_app
    ports:
      - "8000:8000"
    command: ["uv", "run", "hello"]
```

### 2. Optimized Docker Service

Builds and runs the application using the multi-stage Dockerfile.

- Defines the `app-optimized-docker` service, which is built from the `multistage.Dockerfile` in the current directory.
- The `args` section passes build arguments to the `multistage.Dockerfile`.
- The `ports` section maps port 8000 on the host to port 8000 on the container.
- The `command` section specifies the command to run when the container starts.

```yaml
services:
 app-optimized-docker:
    build:
      context: .
      dockerfile: multistage.Dockerfile
      args:
        <<: *default-args
    image: lit_autoencoder_app
    ports:
      - "8000:8000"
    command: ["uv", "run", "hello"]
```

## Build and Run and Test the Docker Application
<!-- Include the content of README.md -->
{%
    include-markdown "../../README.md"
    start="<!--docs-ref-docker-prod-build-start-->"
    end="<!--docs-ref-docker-prod-build-end-->"
%}

## Best Practices

- **Use Multi-Stage Builds:** Multi-stage builds help to reduce the final image size by only including the necessary artifacts.
- **Use `.dockerignore`:** Exclude unnecessary files and directories from the Docker build context to improve build performance and reduce image size.
- **Minimize Layers:** Combine multiple commands into a single `RUN` command to reduce the number of layers in the image.
- **Use Non-Root User:** Run the application as a non-root user to improve security.
- **Use Environment Variables:** Use environment variables to configure the application at runtime.
- **Use a Volume:** Use a volume to persist data between container restarts.

## Getting Started

1. Install Docker Compose: Ensure that Docker Compose is installed on your system. It often comes bundled with Docker Desktop for Mac and Windows.
2. Clone the Repository: Clone the project repository to your local machine, so you have access to the Docker configuration files.
3. Build the Docker Image: Build the Docker image using the provided `Dockerfile` or `multistage.Dockerfile`. To build (or rebuild) your service, use: `docker-compose build`
4. Run the Application: Start the application using Docker Compose. To start your service, use: `docker-compose up`
5. Access the Application: Open your web browser and go to `http://localhost:8000` to view the running application.

## Conclusion

This document provides a comprehensive overview of the Docker production setup used in this project. By following the guidelines and best practices outlined in this document, you can create a lean, efficient, and secure production environment for your application.
