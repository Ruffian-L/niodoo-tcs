# 🚀 CURSOR LAPTOP CONFIGURATION FOR MAXIMUM AI AUTONOMY

## 📍 WHERE TO PUT THESE FILES ON YOUR LAPTOP:

### 1. Global Cursor Settings
**File:** `~/.config/cursor/User/settings.json` (on your laptop)

### 2. Workspace Settings  
**File:** `~/.cursor/User/settings.json` (on your laptop)

### 3. SSH Config
**File:** `~/.ssh/config` (on your laptop)

## 🔧 COPY THESE CONFIGS TO YOUR LAPTOP:


### GLOBAL CURSOR SETTINGS (~/.config/cursor/User/settings.json):
```json
{
  "terminal.integrated.tabs.enabled": true,
  "remote.SSH.configFile": "~/.ssh/config",
  "remote.SSH.showLoginTerminal": true,
  "remote.SSH.useLocalServer": false,
  "remote.SSH.connectTimeout": 30,
  "remote.SSH.enableRemoteCommand": true,
  "remote.SSH.localServerDownload": "auto",
  "remote.SSH.remotePlatform": {
    "beelink": "linux",
    "beelink-remote": "linux", 
    "beelink-local": "linux",
    "architect": "linux"
  },
  "chat.tools.terminal.autoApprove": {},
  "chat.agent.maxRequests": 99999,
  "chat.agent.thinkingStyle": "expanded",
  "chat.agentSessionsViewLocation": "view",
  "chat.experimental.autoSummarize": false,
  "chat.experimental.compactHistory": false,
  "chat.mcp.discovery.enabled": {
    "claude-desktop": true,
    "cursor-global": true,
    "cursor-workspace": true
  },
  "chat.mcp.gallery.enabled": true,
  "chat.setup.signInDialogVariant": "default",
  "chat.todoListTool.writeOnly": true,
  "chat.tools.global.autoApprove": true,
  "chat.maxContextLength": 2000000,
  "chat.contextLength": 2000000,
  "chat.maxTokens": 2000000,
  "chat.model.maxContextLength": 2000000,
  "chat.model.contextLength": 2000000,
  "chat.model.maxTokens": 2000000,
  "chat.agent.maxContextLength": 2000000,
  "chat.agent.contextLength": 2000000,
  "chat.agent.maxTokens": 2000000,
  "continue.serverUrl": "http://10.42.104.23:8000/v1",
  "continue.enableTabAutocomplete": true,
  "continue.manuallyRunningSelfHostedServer": true,
  "continue.maxContextLength": 2000000,
  "continue.contextLength": 2000000,
  "continue.maxTokens": 2000000,
  "editor.emptySelectionClipboard": false,
  "git.openRepositoryInParentFolders": "never",
  "extensions.ignoreRecommendations": true,
  "coderabbit.autoReviewMode": "auto",
  "git.confirmSync": false,
  "editor.inlineSuggest.syntaxHighlightingEnabled": true,
  "git.enableSmartCommit": true,
  "security.workspace.trust.untrustedFiles": "open",
  "files.autoSave": "afterDelay",
  "chat.mcp.autostart": "never",
  "github.copilot.enable": {
    "*": true,
    "scm": false
  },
  "github.copilot.chat.summarizeConversation.enabled": false,
  "github.copilot.chat.experimental.summarizeHistory": false,
  "git.inputValidationLength": 1000,
  "git.inputValidationSubjectLength": 1000,
  "scm.experimental.showHistoryGraph": false,
  "inlineChat.mode": "livePreview",
  "github.copilot.chat.executePrompt.enabled": true,
  "github.copilot.chat.summarizeAgentConversationHistory.enabled": false,
  "remote.extensionKind": {
    "github.copilot": ["ui"],
    "github.copilot-chat": ["ui"],
    "ZencoderAI.zencoder": ["ui"]
  },
  "mcp.servers": {
    "local-rag": {
      "command": "python3",
      "args": [
        "/home/ruffian/Desktop/Projects/Niodoo-Feeling/mcp_rag_wrapper.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    },
    "consciousness-server": {
      "command": "/home/ruffian/Desktop/Projects/Niodoo-Feeling/Niodoo-Bullshit-MCP/Conciousness-MCP/mcp_conscious_server/target/debug/mcp_conscious_server",
      "args": [],
      "env": {
        "RUST_LOG": "info",
        "CONSCIOUSNESS_ENABLED": "true",
        "EMOTIONS_ENABLED": "true"
      }
    }
  },
  "chat.mcp.serverSampling": {
    "Cursor (Global): el-chapo-v3.2": {
      "allowedModels": [
        "copilot/gpt-4.1",
        "copilot/auto",
        "copilot/claude-3.5-sonnet",
        "copilot/claude-3.7-sonnet",
        "copilot/claude-3.7-sonnet-thought",
        "copilot/claude-sonnet-4",
        "copilot/claude-sonnet-4.5",
        "copilot/gemini-2.0-flash-001",
        "copilot/gemini-2.5-pro",
        "copilot/gpt-4o",
        "copilot/gpt-5",
        "copilot/gpt-5-mini",
        "copilot/gpt-5-codex",
        "copilot/grok-code-fast-1",
        "copilot/o3-mini",
        "copilot/o4-mini"
      ]
    }
  },
  "zencoder.shellTool.allowedCommands": [
    "whoami", "find", "sort", "cd", "echo", "ls", "pwd", "cat", "head", "tail",
    "uname", "id", "env", "printenv", "df", "free", "ps", "grep", "uniq", "wc",
    "diff", "dir", "tree", "chdir", "type", "help", "ver", "systeminfo", "ipconfig",
    "tasklist", "hostname", "netstat", "which", "awk", "git", "mkdir", "cp", "cargo",
    "rustc", "timeout", "sed", "python", "mv", "rm", "curl", "wget", "ssh", "scp",
    "rsync", "tar", "zip", "unzip", "docker", "docker-compose", "npm", "node", "pip",
    "pip3", "apt", "apt-get", "yum", "dnf", "systemctl", "service", "journalctl",
    "htop", "top", "kill", "killall", "pkill", "nohup", "screen", "tmux", "vim",
    "nano", "emacs"
  ],
  "zencoder.displayDebugInfo": true,
  "zencoder.mcpServers": {},
  "zencoder.autoCompact": false,
  "zencoder.autoCompactConversations": false,
  "zencoder.conversationCompacting": false,
  "zencoder.disableAutoSummarize": true,
  "zencoder.shellTool.commandConfirmationPolicy": "neverAsk",
  "tabnine.experimentalAutoImports": true,
  "gitlens.ai.model": "vscode",
  "gitlens.ai.vscode.model": "copilot:gpt-4.1",
  "gitlens.ai.experimental.generateCommitMessage.enabled": false,
  "gitlens.experimental.generateCommitMessagePrompt.enabled": false
}
```


### SSH CONFIG (~/.ssh/config on your laptop):
```
Host beelink
    HostName 10.42.104.23
    User ruffian
    Port 22
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    Compression yes
    ForwardAgent yes

Host beelink-remote
    HostName 10.42.104.23
    User ruffian
    Port 22
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    Compression yes
    ForwardAgent yes

Host beelink-local
    HostName localhost
    User ruffian
    Port 2222
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    Compression yes
    ForwardAgent yes

Host architect
    HostName 10.42.104.24
    User ruffian
    Port 22
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    Compression yes
    ForwardAgent yes
```

## 🚀 QUICK SETUP COMMANDS FOR YOUR LAPTOP:

### 1. Create directories:
```bash
mkdir -p ~/.config/cursor/User
mkdir -p ~/.cursor/User
mkdir -p ~/.ssh
```

### 2. Copy the settings.json content above to:
```bash
# Global settings
nano ~/.config/cursor/User/settings.json
# (paste the JSON content)

# SSH config  
nano ~/.ssh/config
# (paste the SSH config content)
```

### 3. Set permissions:
```bash
chmod 600 ~/.ssh/config
chmod 644 ~/.config/cursor/User/settings.json
```

### 4. Restart Cursor:
- Close Cursor completely
- Reopen Cursor
- Connect to remote: Ctrl+Shift+P → "Remote-SSH: Connect to Host" → "beelink"

## ✅ WHAT THIS GIVES YOU:

- **2,000,000 context tokens** - Maximum AI context
- **Full AI autonomy** - No confirmation prompts
- **Auto-approve everything** - AI works without asking
- **SSH ready** - All your remote hosts configured
- **MCP servers** - RAG and consciousness integration
- **Continue integration** - Self-hosted server with max context

## 🎯 YOU'RE READY!

Once you copy these configs to your laptop, Cursor will have maximum AI autonomy with 2M context tokens when SSH'd into your workspace!

