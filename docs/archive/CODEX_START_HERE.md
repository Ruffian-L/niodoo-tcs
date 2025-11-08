# CODEX: START HERE

**Your task:** Integrate Niodoo consciousness system into this repo.

---

## EXACT COMMANDS TO RUN

### PHASE 1: Copy Files (30 min)

```bash
cd /home/ruffian/Desktop/Niodoo-Final
./CODEX_EXACT_COPY_COMMANDS.sh
```

**Expected output:**
```
=== PHASE 1: COPYING NIODOO FILES TO NIODOO-FINAL ===
Step 1.1: Creating niodoo-core directory structure...
Step 1.2: Copying core consciousness files...
...
=== PHASE 1 COMPLETE ===
```

**Then report:** "✅ PHASE 1 complete - 36 files copied"

---

### PHASE 2: Create Cargo.toml (5 min)

```bash
cd /home/ruffian/Desktop/Niodoo-Final
./CODEX_PHASE_2_CARGO.sh
```

**Expected output:**
```
=== PHASE 2: CREATING NIODOO-CORE CARGO.TOML ===
  ✓ Created niodoo-core/Cargo.toml
  ✓ Created niodoo-core/src/lib.rs
=== PHASE 2 COMPLETE ===
```

**Then report:** "✅ PHASE 2 complete - niodoo-core Cargo.toml created"

---

### PHASE 3: Update Workspace (5 min)

```bash
cd /home/ruffian/Desktop/Niodoo-Final
./CODEX_PHASE_3_WORKSPACE.sh
```

**Expected output:**
```
=== PHASE 3: UPDATING ROOT CARGO.TOML ===
  ✓ Added niodoo-core to workspace members
  ✓ Verification passed: niodoo-core found in Cargo.toml
  Running cargo check to verify workspace...
=== PHASE 3 COMPLETE ===
```

**Then report:** "✅ PHASE 3 complete - workspace updated"

---

## AFTER PHASE 3

**STOP and report back:**
```
✅ PHASE 1 complete - Files copied
✅ PHASE 2 complete - Cargo.toml created
✅ PHASE 3 complete - Workspace updated

cargo check output: [paste output]
```

**Then Friend Claude will:**
1. Verify it's working
2. Create Phase 4 script (integration module)
3. Guide you through final phases

---

## IF ERRORS OCCUR

**Report exactly:**
1. Which phase failed
2. The exact error message
3. The command that failed

**Do NOT:**
- Try to fix it yourself
- Skip steps
- Modify paths

---

## FILES LOCATIONS REFERENCE

```
SOURCE: /home/ruffian/Desktop/Projects/Niodoo-Feeling/
DEST:   /home/ruffian/Desktop/Niodoo-Final/
```

---

**Ready? Run Phase 1 first.**
