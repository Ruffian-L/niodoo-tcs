# NIODOO Mind's Eye Web Visualizer

Web-based real-time 3D visualization of NIODOO cognitive state using Three.js. **Perfect for SSH/RunPod scenarios** - no GPU or X11 needed!

## Overview

The Mind's Eye Visualizer serves a web page that connects to the NIODOO telemetry stream and renders the AI's cognitive state in real-time:
- **TwistedTorus mesh** - The cognitive manifold as a 3D torus surface
- **Consciousness point** - A glowing spark showing the current state position
- **Betti number visualizations** - Topological features (fragmentation via scaling)
- **Compass quadrant tinting** - Scene color changes based on cognitive state (Panic/Discover/Persist/Master)
- **Real-time metrics** - Side panel showing iteration, Betti numbers, PAD state, etc.

## Prerequisites

- Rust 1.70+ with Cargo
- NIODOO running with telemetry enabled
- Web browser (Chrome/Firefox/Safari)

## Building

```bash
cargo build --release
```

## Running on RunPod/SSH

### Option 1: Port Forwarding (Recommended)

1. **On your local machine**, set up SSH port forwarding:
   ```bash
   ssh -L 8080:localhost:8080 user@your-runpod-ip
   ```

2. **On RunPod**, start NIODOO with telemetry:
   ```bash
   NIODOO_TELEMETRY_ENABLED=true cargo run --bin niodoo_real_integrated
   ```

3. **On RunPod**, start the visualizer:
   ```bash
   cargo run --bin niodoo-visualizer -- --port 8080
   ```

4. **On your local machine**, open browser:
   ```
   http://localhost:8080
   ```

### Option 2: Public URL (if RunPod allows)

If your RunPod has a public URL/port exposed:
1. Start visualizer: `cargo run --bin niodoo-visualizer -- --port 8080`
2. Open `http://your-runpod-public-ip:8080` in browser

## Configuration

- **Visualizer Port**: Default is `8080`. Change with `--port` flag
- **Telemetry Port**: Default is `9999`. Change via `NIODOO_TELEMETRY_PORT` env var in NIODOO

## Troubleshooting

- **Connection failed**: Ensure NIODOO is running with `NIODOO_TELEMETRY_ENABLED=true`
- **No updates**: Check that NIODOO is processing prompts (telemetry only broadcasts during prompt processing)
- **Can't access from browser**: Use SSH port forwarding (`ssh -L 8080:localhost:8080 ...`)
- **WebSocket connection fails**: Check firewall/port settings

## Features

- ✅ Works over SSH (no X11/GPU needed)
- ✅ Real-time 3D visualization with Three.js
- ✅ Automatic reconnection
- ✅ Responsive UI with live metrics
- ✅ Compass quadrant color coding
- ✅ Betti number visual effects

