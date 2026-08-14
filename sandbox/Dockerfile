# MyCoder sandbox image.
#
# Pinned to python:3.12-slim so the container ABI matches the host venv
# (Python 3.12.0). The same image tag must satisfy both sides of the
# host/container boundary or tool output (pickle, __pycache__, .pyc) can
# silently mismatch.
#
# Two security layers live HERE vs. at RUNTIME (sandbox/docker_executor.py):
#   - this file builds the image (tools + a non-root user)
#   - the executor starts every container with read_only rootfs, no network,
#     dropped capabilities, no-new-privileges, mem/cpu caps, and the project
#     mounted read-only at /src. Keeping the runtime constraints in code (not
#     baked into the image) lets them be enforced per-launch and audited.

FROM python:3.12-slim

# --- tooling ---------------------------------------------------------------
# git + ripgrep are the agent's bread and butter; rsync is used for
# workspace<->host sync tooling; curl/gzip exist only to fetch the
# tree-sitter-cli binary and are part of the image for reproducibility.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ripgrep \
        rsync \
        curl \
        gzip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# tree-sitter-cli is a Rust binary that is NOT published on PyPI. We pin the
# official GitHub release binary rather than pull in node+npm just for it
# (npm's copy is the same artifact; the direct download keeps the image ~100MB
# slimmer). Version is pinned for reproducible builds, and the architecture is
# selected via Docker's TARGETARCH build arg so the same Dockerfile builds a
# working image on both amd64 (CI) and arm64 (Apple Silicon) hosts — pulling
# the x64 binary on arm64 would otherwise die under Rosetta with a missing
# x86_64 loader.
ARG TREE_SITTER_VERSION=v0.25.7
ARG TARGETARCH
# case/if rather than bash's ${VAR/amd64/x64}: the base image's /bin/sh is
# dash (POSIX), which rejects bash-only substitution.
RUN case "${TARGETARCH}" in \
      amd64) ARCH="x64" ;; \
      *)     ARCH="${TARGETARCH:-x64}" ;; \
    esac \
    && curl -fsSL \
        "https://github.com/tree-sitter/tree-sitter/releases/download/${TREE_SITTER_VERSION}/tree-sitter-linux-${ARCH}.gz" \
      | gunzip > /usr/local/bin/tree-sitter \
    && chmod +x /usr/local/bin/tree-sitter \
    && tree-sitter --version

# --- non-root runtime user -------------------------------------------------
# The agent's commands run as this user, never as root. `nologin` shell: the
# sandbox never needs interactive logins. `--user-group` creates a matching
# group too (a bare --gid 1000 would fail: a fresh slim image has no group
# 1000). The Dockerfile pre-creates /workspace and chowns it to sandbox so
# that a fresh named volume mounted there inherits sandbox ownership from the
# image — which is what lets the executor provision it WITHOUT running as root.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin --user-group sandbox \
    && mkdir -p /workspace /src \
    && chown -R sandbox:sandbox /workspace

USER sandbox
WORKDIR /workspace

# The executor overrides the command with ["sleep","infinity"]; kept here so a
# bare `docker run` of this image also stays alive and inspectable.
CMD ["sleep", "infinity"]
