#!/bin/bash
# EMERGENCY VS CODE SSH FIX SCRIPT
# This will add Remote-SSH configuration to your VS Code settings

SETTINGS_FILE="$HOME/.config/Code/User/settings.json"

echo "🔥 FIXING VS CODE SSH CONFIG NOW..."

# Backup first
cp "$SETTINGS_FILE" "$SETTINGS_FILE.backup.$(date +%s)"
echo "✅ Backup created"

# Use sed to insert the Remote-SSH config after line 2
sed -i '2 a\  "remote.SSH.configFile": "/home/ruffian/.ssh/config",\n  "remote.SSH.showLoginTerminal": true,\n  "remote.SSH.useLocalServer": false,\n  "remote.SSH.connectTimeout": 30,\n  "remote.SSH.remotePlatform": {\n    "beelink": "linux",\n    "beelink-remote": "linux",\n    "beelink-local": "linux",\n    "architect": "linux"\n  },' "$SETTINGS_FILE"

echo "✅ VS CODE SETTINGS UPDATED!"
echo ""
echo "NOW DO THIS:"
echo "1. Close all VS Code windows"
echo "2. Reopen VS Code"
echo "3. Press Ctrl+Shift+P"
echo "4. Type: Remote-SSH: Connect to Host"
echo "5. Select: beelink"
echo ""
echo "🚀 YOU'RE READY TO GO BRO!!"