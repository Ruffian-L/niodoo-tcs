#!/bin/bash
# Start telemetry bridge for visualization
# This bridges TCP telemetry (port 9999) to WebSocket (port 8765) and serves HTML (port 8080)

cd "$(dirname "$0")"
python3 simple_bridge.py

