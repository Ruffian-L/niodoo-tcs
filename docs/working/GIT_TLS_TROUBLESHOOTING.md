# Git TLS Handshake Troubleshooting for Remote Deployment

**Problem**: Cargo may fail to fetch git dependencies on remote host due to TLS handshake issues.

**Impact**: Blocks the entire integration build if git dependencies can't be resolved.

---

## Solution 1: Vendor Dependencies (RECOMMENDED)

Bundle all dependencies locally to avoid git fetches during build.

**Setup**: Before running these commands, set your project directory path:
```bash
# Example setup (adjust paths to match your environment):
export LOCAL_PROJECT_DIR=~/Desktop/Niodoo-Final
export REMOTE_PROJECT_DIR=~/Desktop/Niodoo-Final
```

### On Laptop (where it builds successfully):
```bash
cd $LOCAL_PROJECT_DIR

# Generate vendored dependencies
cargo vendor > .cargo/config.toml.vendor

# This creates a vendor/ directory with all deps
# Syncthing will auto-sync this to remote host
```

### On Remote Host (after Syncthing sync):
```bash
cd $REMOTE_PROJECT_DIR

# Copy vendor config to active cargo config
cp .cargo/config.toml.vendor .cargo/config.toml

# Now cargo build uses local vendor/ directory
cargo build --release --offline
```

**Advantages:**
- No network required during build
- No TLS handshake issues
- Reproducible builds
- Faster builds (no re-downloading)

**Disadvantages:**
- vendor/ directory is large (~500MB-1GB)
- Must re-vendor if dependencies change

---

## Solution 2: Git Configuration Workarounds

### Option A: Use system git instead of libgit2
```bash
# On remote host:
export CARGO_NET_GIT_FETCH_WITH_CLI=true
cargo build --release
```

This forces cargo to use system git (which may have better TLS config) instead of embedded libgit2.

### Option B: Disable TLS verification (INSECURE - testing only)

**IMPORTANT**: Apply only to the current repository (--local) or a single command, NOT globally:

```bash
# Option B1: Repository-scoped (preferred for testing)
# On remote host, in the project directory:
git config --local http.sslVerify false
cargo build --release

# REVERT IMMEDIATELY AFTER:
git config --local --unset http.sslVerify
```

OR

```bash
# Option B2: Single-command scoped (even safer)
# On remote host:
git -c http.sslVerify=false clone <repo_url>
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo build --release
```

**Only use this to diagnose if TLS is the actual problem. DO NOT ship with this config.**

### Option C: Update CA certificates
```bash
# On remote host:
sudo apt update
sudo apt install ca-certificates
sudo update-ca-certificates
```

---

## Solution 3: HTTP instead of HTTPS (if safe)

If you control the git server (e.g., local Gitea):

```toml
# In Cargo.toml, replace:
niodoo-core = { git = "https://gitea:3000/..." }

# With (replace <GIT_SERVER_IP> with your actual server IP):
niodoo-core = { git = "http://<GIT_SERVER_IP>:3000/..." }
```

**Only safe for:**
- Local network (Tailscale)
- Private git server (Gitea on your server)
- No sensitive credentials in transit

---

## Solution 4: SSH instead of HTTPS

```toml
# In Cargo.toml, replace:
niodoo-core = { git = "https://github.com/..." }

# With:
niodoo-core = { git = "ssh://git@github.com/..." }
```

Requires SSH keys configured on remote host.

---

## Quick Diagnosis

### Test if git TLS works on remote host:
```bash
ssh -i <SSH_KEY_PATH> <USER>@<HOST_IP>

# On remote host:
git clone https://github.com/rust-lang/cargo.git /tmp/test-tls
# If this succeeds: TLS is fine, problem is elsewhere
# If this fails: TLS is broken, use Solution 1 or 2
```

### Check current cargo config:
```bash
# On remote host:
cat ~/.cargo/config.toml
env | grep -i cargo
```

---

## Recommended Deployment Strategy

**Phase 1: Quick Test (use vendoring)**
1. Vendor on laptop: `cargo vendor`
2. Sync to remote host via Syncthing
3. Build offline: `cargo build --offline`
4. This proves the code works, independent of network

**Phase 2: Production (fix TLS properly)**
1. Diagnose TLS issue (test git clone)
2. Update CA certs if needed
3. Use system git if libgit2 is broken
4. Keep vendored deps as backup

---

## Remote Host Environment Notes

**Known Working:**
- CUDA 12.8 ✅
- PyTorch 2.8.0+cu128 ✅
- RTX Quadro 6000 48GB ✅
- ONNX Runtime ✅

**Needs Verification:**
- Git TLS handshake ⚠️
- Model symlinks (after move from laptop) ⚠️
- Syncthing sync status ⚠️

---

## Integration Prompt Update

**For Grok:** Before attempting `cargo build`, run quick TLS test:

```bash
# Test 1: Can git clone over HTTPS?
git clone https://github.com/rust-lang/cargo.git /tmp/test-tls && rm -rf /tmp/test-tls

# Test 2: If fails, use vendored dependencies
if [ -d vendor ]; then
    echo "Using vendored dependencies (offline build)"
    cargo build --release --offline
else
    echo "ERROR: vendor/ directory not found and git TLS broken"
    echo "Run 'cargo vendor' on laptop and sync via Syncthing"
    exit 1
fi
```

---

## Fallback Plan

If ALL solutions fail:

1. Build on laptop (where it works)
2. Copy binary to remote host:
   ```bash
   # On laptop:
   cargo build --release
   scp -i <PATH_TO_SSH_KEY> \
       target/release/niodoo_consciousness \
       <USER>@<BEELINK_HOST>:/home/<USER>/bin/
   ```
3. Run binary directly on remote host (no build needed)

**Trade-off:** Loses GPU-specific optimizations from native compilation.

---

## Status Tracking

- [ ] Test git TLS on remote host
- [ ] Vendor dependencies on laptop
- [ ] Verify Syncthing synced vendor/
- [ ] Attempt offline build
- [ ] Update CA certs if needed
- [ ] Configure CARGO_NET_GIT_FETCH_WITH_CLI if needed

---

**Next Action:** Test git TLS handshake on remote host to confirm diagnosis, then choose solution path.
