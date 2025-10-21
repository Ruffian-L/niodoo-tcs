#!/bin/bash

# Set environment variables to help ring compilation
export CC=gcc
export CXX=g++
export AR=ar
export RANLIB=ranlib
export RING_BUILDING=1

# Clean the build
cargo clean

# Build with verbose output to debug linking
CARGO_BUILD_JOBS=1 cargo build --bin rut_gauntlet -vv