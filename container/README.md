# Container

## Docker

- `docker-compose.yaml` support `${VAR-default}` which mean that use default if no VAR. `default` can be string like `'string'`
- `docker-compose.yaml` relies on `.env` file or shell environment

### Volumes
- `docker-compose.yaml` use `volumes` for persist data, can define a named volume in the `volumes` section
    - **Persists Data**: Named volumes ensure that data written by the containers to specific paths is persisted outside the containers and remains intact across container restarts or deletions.
    - **Separation**: By defining these volumes separately, each container can have isolated storage managed by Docker.
    - **Sharing Across Containers**: Named volumes can be shared between containers, allowing them to read/write the same data if needed.
    - **Empty Name volume**, when define a volume using `{}`, you're essentially creating a named volume with default settings.
        1. **Named Volume Creation**: Docker Compose will create a named volume called `ollama`. This named volume is managed by Docker and will be used to persist data written to it by the container.
        2. **Default Location**:
            - On Linux systems, Docker typically stores named volume data inside `/var/lib/docker/volumes/`. Each volume has a subdirectory within this path.
            - On Windows or macOS, the location might vary depending on the Docker Desktop configuration, but it's generally managed internally by Docker.
        3. **Data Management**: Docker handles the storage and lifecycle of this volume, ensuring data is retained across container restarts or removals.
        4. **Named Volumes** use `{}` without host paths. They're managed by Docker.
        5. **Bind Mounts** include paths and are specified directly in the **service’s `volumes` section**.

### Config Override
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d` can make dev override base config

### Pass build arg to Dockerfile
- Add `args` under `build` to specify build args, Add `ARG` in Dockerfile

### Commands
- `cp`: Copy data from/to docker
- `compose start/stop/up/down`: **up** will build if have build config, **start** always run in detached mode
- `compose up --build`: explicitly specify build to trigger rebuild
- `compose down`: when environment variable change

### Multiple Stage Build

1. **Global `ARG` Declaration**:
   - You can declare `ARG` variables globally, such that they are available in all build stages. If you declare the `ARG` before any `FROM` instructions, it is accessible in all stages:
        ```Dockerfile
        ARG GLOBAL_ARG
        FROM node:14 AS builder
        RUN echo "Global ARG in first stage: $GLOBAL_ARG"

        FROM node:14 AS final
        RUN echo "Global ARG in second stage: $GLOBAL_ARG"
        ```

        Declaring `ARG GLOBAL_ARG` outside any specific stage makes it available for both `builder` and `final` stages.

2. **Stage-Specific `ARG` Declaration**:
   - If you declare `ARG` after a `FROM` instruction, it only applies to the immediate stage following that declaration.
        ```Dockerfile
        FROM node:14 AS builder
        ARG STAGE_ARG
        RUN echo "Stage ARG in builder: $STAGE_ARG"

        FROM node:14 AS final
        ARG ANOTHER_STAGE_ARG
        RUN echo "Stage ARG in final: $ANOTHER_STAGE_ARG"
        ```
        
        Here, `STAGE_ARG` is only available in the `builder` stage, and `ANOTHER_STAGE_ARG` is only available in the `final` stage.
