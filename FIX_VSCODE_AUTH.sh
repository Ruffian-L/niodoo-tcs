#!/bin/bash
# Fix VS Code extension authentication over SSH

SETTINGS="/home/ruffian/.config/Code/User/settings.json"

echo "🔧 Fixing Copilot & Zencoder authentication for Remote-SSH..."

# Backup first
cp "$SETTINGS" "$SETTINGS.backup.auth.$(date +%s)"
echo "✅ Backup created"

# Remove the last closing brace, add our settings, then close again
sed -i '$ d' "$SETTINGS"

cat >> "$SETTINGS" << 'EOF'
  "remote.extensionKind": {
    "github.copilot": ["ui"],
    "github.copilot-chat": ["ui"],
    "ZencoderAI.zencoder": ["ui"]
  },
  "remote.SSH.enableRemoteCommand": true,
  "remote.SSH.localServerDownload": "auto"
}
EOF

echo "✅ Settings updated!"
echo ""
echo "Now do this:"
echo "1. Close ALL VS Code windows"
echo "2. Reopen VS Code"
echo "3. Connect to beelink via SSH"
echo "4. Try Copilot Chat or Zencoder - should work now!"
echo ""
echo "If it STILL doesn't work, manually sign out and sign back in:"
echo "  - Copilot: Click account icon → Sign out → Sign in"
echo "  - Zencoder: Ctrl+Shift+P → 'Zencoder: Sign Out' → Sign in again"