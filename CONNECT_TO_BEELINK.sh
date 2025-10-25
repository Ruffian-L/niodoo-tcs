#!/bin/bash
# Quick connect to Beelink and open Niodoo workspace

echo "🚀 Connecting to Beelink..."
echo ""
echo "BEELINK INFO:"
ssh beelink "echo '  Hostname: ' && hostname && echo '  GPU: ' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && echo '  Rust: ' && rustc --version && echo '  Path: ' && pwd"
echo ""
echo "✅ Connection verified!"
echo ""
echo "To open in VS Code Remote:"
echo "  1. Press Ctrl+Shift+P"
echo "  2. Type: Remote-SSH: Connect to Host"
echo "  3. Select: beelink"
echo "  4. Open folder: /home/beelink/Niodoo-Final"
echo ""
echo "Or SSH directly:"
echo "  ssh beelink"