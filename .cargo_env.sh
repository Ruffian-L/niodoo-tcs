#!/bin/bash
# Cargo environment configuration to use /workspace instead of /tmp
export CARGO_TARGET_DIR=/workspace/Niodoo-Final/target
export TMPDIR=/workspace/Niodoo-Final/tmp
export CCACHE_DIR=/workspace/Niodoo-Final/.ccache
mkdir -p "$TMPDIR" "$CCACHE_DIR"
