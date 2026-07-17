# Stage 1: build
#
# rust:1.94-bookworm (not -slim, not trixie):
#  - 1.94 because the workspace is edition 2024 and turbomcp declares rust-version 1.89.
#  - the buildpack-deps base already ships `git`, which the turbomcp git dependency needs.
#  - -bookworm so the produced binary's glibc matches the bookworm-slim runtime below. The bare
#    `rust:1.94` / `-slim` tags are trixie-based (glibc 2.38) and the binary then dies on bookworm
#    with "GLIBC_2.38 not found". Bump both stages together or not at all.
FROM rust:1.94-bookworm AS builder
WORKDIR /build

# fastembed pulls ort (ONNX Runtime), which needs a C++ toolchain and libssl headers to build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev cmake clang \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole workspace: path dependencies (lqm-core/lqm-ingest) mean a manifest-only
# dependency-warmup layer would need every member's Cargo.toml anyway, for little cache gain.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Only lqm-mcp is deployed. lqm-cli/lqm-api are operator tools and building them here would drag in
# extra dependencies for no benefit to this image.
RUN cargo build --release -p lqm-mcp

# Stage 2: runtime
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="liberado-qdrant-mcp"
LABEL org.opencontainers.image.description="MCP server for RAG over Qdrant with local ONNX embeddings"
LABEL org.opencontainers.image.source="https://github.com/ForrestThump/liberado-qdrant-mcp"
LABEL org.opencontainers.image.licenses="MIT"

# curl is required by the compose healthcheck; ca-certificates for fetching the ONNX model.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/lqm-mcp /usr/local/bin/lqm-mcp

# fastembed downloads its ONNX model on first use and caches it relative to the working directory.
# Mount a volume here or every restart re-downloads it.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN mkdir -p /app/.fastembed_cache

# Serve HTTP by default. With no subcommand the binary would take the STDIO transport, which is
# useless in a container Liberado reaches over the network.
EXPOSE 3000
ENTRYPOINT ["lqm-mcp"]
CMD ["serve", "--host", "0.0.0.0", "--port", "3000"]
