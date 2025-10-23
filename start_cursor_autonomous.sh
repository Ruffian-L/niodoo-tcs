#!/bin/bash
# Cursor Autonomous Startup Script
# Maximum AI autonomy with 2M context tokens

echo "🚀 Starting Cursor with Maximum AI Autonomy..."

# Set environment variables for maximum AI performance
export CURSOR_MAX_CONTEXT=2000000
export CURSOR_AI_AUTONOMY=true
export CURSOR_AUTO_APPROVE=true
export CURSOR_MCP_ENABLED=true
export CURSOR_THINKING_STYLE=expanded

# Ensure Cursor config directories exist
mkdir -p ~/.config/cursor/User
mkdir -p ~/.cursor/User
mkdir -p /workspace/Niodoo-Final/.cursor

# Set proper permissions
chmod 755 ~/.config/cursor
chmod 644 ~/.config/cursor/User/settings.json
chmod 644 ~/.cursor/User/keybindings.json
chmod 644 /workspace/Niodoo-Final/.cursor/settings.json

# Start Cursor with no-sandbox flag (required for Linux)
echo "🔥 Launching Cursor with 2M context tokens and full AI autonomy..."
cursor --no-sandbox /workspace/Niodoo-Final &

echo "✅ Cursor launched with maximum AI autonomy!"
echo "📊 Context tokens: 2,000,000"
echo "🤖 AI autonomy: FULL"
echo "🔧 MCP servers: ENABLED"
echo "⚡ Auto-approve: ENABLED"
echo ""
echo "Ready to code with maximum AI power! 🚀"
