# CLUSTER MAP - READ THIS FIRST

## 🖥️ YOU ARE HERE (CURRENT MACHINE)
**Hostname:** `enochsawakening` (LAPTOP - Developer node)
**User:** `ruffian`
**Repo Path:** `/home/ruffian/Desktop/Niodoo-Final`
**Hardware:** RTX 5080 16GB
**Tailscale IP:** `100.126.84.41`

---

## 🌐 CLUSTER NODES

### 🏗️ ARCHITECT (Beelink Server)
- **Hostname:** `beelink`
- **User:** `beelink` 
- **Tailscale IP:** `100.113.10.90`
- **Local IP:** `10.42.104.23` (2.5GbE network)
- **Hardware:** RTX A6000 48GB VRAM
- **SSH:** `ssh beelink` (uses `~/.ssh/temp_beelink_key`)
- **Repo Paths:**
  - `/home/beelink/Niodoo-Final` ← MAIN PRODUCTION
  - `/home/beelink/Niodoo-Final-remote`
  - `/home/beelink/Niodoo-Feeling`
  - `/home/beelink/Niodoo-Bullshit-MCP`

### 💼 WORKER (Old Laptop)
- **Hostname:** `oldlaptop`
- **User:** `oldlaptop`
- **Tailscale IP:** `100.119.255.24`
- **Local IP:** `10.42.104.223`
- **Hardware:** Intel Ultra 5 (CPU only)
- **SSH:** `ssh oldlaptop` (uses `~/.ssh/id_oldlaptop`)

---

## 🔧 QUICK COMMANDS

### Test Connection to Beelink:
```bash
ssh beelink whoami  # Should output: beelink
```

### Run on Beelink (from laptop):
```bash
ssh beelink "cd ~/Niodoo-Final && cargo build --all"
```

### VS Code Remote SSH:
1. Press `Ctrl+Shift+P`
2. Type "Remote-SSH: Connect to Host"
3. Select `beelink`
4. Open folder: `/home/beelink/Niodoo-Final`

---

## ⚠️ IMPORTANT FOR AI ASSISTANTS

**WHEN USER SAYS "on laptop/local":**
- Path: `/home/ruffian/Desktop/Niodoo-Final`
- User: `ruffian`
- Machine: `enochsawakening`

**WHEN USER SAYS "on server/beelink":**
- Path: `/home/beelink/Niodoo-Final`
- User: `beelink`
- Machine: `beelink`
- Connect via: `ssh beelink "command"`

**CURRENT SESSION:**
You are running commands on the **LAPTOP** by default.
To run on Beelink, prefix with: `ssh beelink "..."`