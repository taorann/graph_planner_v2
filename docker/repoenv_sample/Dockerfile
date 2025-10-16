# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential rsync \
    && rm -rf /var/lib/apt/lists/*

# Pre-populate a sample repository that the agent can interact with.
WORKDIR /opt/sample_repo
COPY sample_repo/ /opt/sample_repo/

RUN git init \
    && git config user.email "repoenv@example.com" \
    && git config user.name "RepoEnv Sample" \
    && git add . \
    && git commit -m "initial failing state" \
    && mkdir -p /testbed \
    && rsync -a --delete ./ /testbed/ \
    && cd /testbed \
    && git init \
    && git config user.email "repoenv@example.com" \
    && git config user.name "RepoEnv Sample" \
    && git add . \
    && git commit -m "initial failing state" \
    && pip install --no-cache-dir pytest

WORKDIR /testbed

# Provide a convenient command for manual debugging.
CMD ["bash"]
