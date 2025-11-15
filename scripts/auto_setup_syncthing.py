#!/usr/bin/env python3
"""Automatically configure Syncthing pairing and folders via REST API"""
import json
import sys
import subprocess
import requests

LOCAL_API_KEY = subprocess.check_output(
    "sudo grep -oP '(?<=<apikey>)[^<]+' /root/.local/state/syncthing/config.xml | head -1",
    shell=True, text=True
).strip()

LOCAL_DEVICE_ID = "6MK4WX6-OCG75KW-RRFKSCO-RIH5IP2-2BIB5VG-3QKIIPK-R3J65DI-B2YKCAM"
RUNPOD_DEVICE_ID = "WJHFRVE-6FGGCZW-TOJ4Q3N-6WBSJSQ-DTFO6XC-ZXC3DZ6-OBEFSTL-YOMC4Q4"
RUNPOD_API_KEY = "AjhwJyc2JMfMWhvHmhPz659WF3NibxTD"

LOCAL_URL = "http://localhost:8384"
RUNPOD_URL = "http://38.80.152.72:8384"  # Will use SSH tunnel

headers = {"X-API-Key": LOCAL_API_KEY}

print("=== Auto-configuring Syncthing ===")

# Step 1: Get local config and add RunPod device
print("\n1. Adding RunPod device to local Syncthing...")
config = requests.get(f"{LOCAL_URL}/rest/system/config", headers=headers).json()

# Check if RunPod device already exists
device_exists = any(d['deviceID'] == RUNPOD_DEVICE_ID for d in config.get('devices', []))
if not device_exists:
    config['devices'].append({
        "deviceID": RUNPOD_DEVICE_ID,
        "name": "RunPod",
        "addresses": ["dynamic"],
        "compression": "metadata",
        "introducedBy": "",
        "paused": False,
        "allowedNetworks": [],
        "autoAcceptFolders": False,
        "maxSendKbps": 0,
        "maxRecvKbps": 0,
        "ignoredFolders": [],
        "maxRequestKiB": 0,
        "untrusted": False,
        "remoteGUIPort": 0
    })
    resp = requests.put(f"{LOCAL_URL}/rest/system/config", headers=headers, json=config)
    print(f"   ✓ Added RunPod device: {resp.status_code}")
else:
    print("   ✓ RunPod device already exists")

# Step 2: Add local device to RunPod (via SSH)
print("\n2. Adding local device to RunPod Syncthing...")
ssh_cmd = f'''ssh -T root@38.80.152.72 -p 30572 -i ~/.ssh/id_ed25519 "curl -s -H 'X-API-Key: {RUNPOD_API_KEY}' http://localhost:8384/rest/system/config"'''
runpod_config_json = subprocess.check_output(ssh_cmd, shell=True, text=True)
runpod_config = json.loads(runpod_config_json)

device_exists = any(d['deviceID'] == LOCAL_DEVICE_ID for d in runpod_config.get('devices', []))
if not device_exists:
    runpod_config['devices'].append({
        "deviceID": LOCAL_DEVICE_ID,
        "name": "Local-Machine",
        "addresses": ["dynamic"],
        "compression": "metadata",
        "introducedBy": "",
        "paused": False,
        "allowedNetworks": [],
        "autoAcceptFolders": False,
        "maxSendKbps": 0,
        "maxRecvKbps": 0,
        "ignoredFolders": [],
        "maxRequestKiB": 0,
        "untrusted": False,
        "remoteGUIPort": 0
    })
    config_json = json.dumps(runpod_config)
    ssh_cmd = f'''ssh -T root@38.80.152.72 -p 30572 -i ~/.ssh/id_ed25519 "curl -s -X PUT -H 'X-API-Key: {RUNPOD_API_KEY}' -H 'Content-Type: application/json' -d '{config_json}' http://localhost:8384/rest/system/config"'''
    result = subprocess.check_output(ssh_cmd, shell=True, text=True)
    print(f"   ✓ Added local device to RunPod")
else:
    print("   ✓ Local device already exists on RunPod")

# Step 3: Create folders on RunPod
print("\n3. Creating folders on RunPod...")

# Check if models folder exists
models_folder_exists = any(f['id'] == 'runpod-models' for f in runpod_config.get('folders', []))
if not models_folder_exists:
    runpod_config['folders'].append({
        "id": "runpod-models",
        "label": "RunPod Models",
        "filesystemType": "basic",
        "path": "/workspace/models",
        "type": "sendreceive",
        "devices": [{"deviceID": LOCAL_DEVICE_ID, "introducedBy": "", "encryptionPassword": ""}],
        "rescanIntervalS": 3600,
        "fsWatcherEnabled": True,
        "fsWatcherDelayS": 10,
        "ignorePerms": False,
        "autoNormalize": True,
        "minDiskFree": {"value": 1, "unit": "%"},
        "versioning": {"type": "", "params": {}, "cleanupIntervalS": 3600},
        "copiers": 0,
        "pullerMaxPendingKiB": 0,
        "hashers": 0,
        "order": "random",
        "pullerPauseS": 0,
        "maxConflicts": 10,
        "disableSparseFiles": False,
        "disableTempIndexes": False,
        "paused": False,
        "weakHashThresholdPct": 25,
        "markerName": ".stfolder",
        "copyOwnershipFromParent": False,
        "modTimeWindowS": 0,
        "maxConcurrentWrites": 2,
        "disableFsync": False,
        "blockPullOrder": "standard",
        "copyRangeMethod": "standard",
        "caseSensitiveFS": False,
        "junctionsAsDirs": False,
        "syncOwnership": False,
        "sendOwnership": False,
        "syncXattrs": False,
        "sendXattrs": False,
        "xattrFilter": {"entries": None, "regexes": None}
    })
    print("   ✓ Added models folder config")

# Check if Niodoo-Final folder exists
niodoo_folder_exists = any(f['id'] == 'niodoo-final' for f in runpod_config.get('folders', []))
if not niodoo_folder_exists:
    runpod_config['folders'].append({
        "id": "niodoo-final",
        "label": "Niodoo-Final",
        "filesystemType": "basic",
        "path": "/workspace/Niodoo-Final",
        "type": "sendreceive",
        "devices": [{"deviceID": LOCAL_DEVICE_ID, "introducedBy": "", "encryptionPassword": ""}],
        "rescanIntervalS": 3600,
        "fsWatcherEnabled": True,
        "fsWatcherDelayS": 10,
        "ignorePerms": False,
        "autoNormalize": True,
        "minDiskFree": {"value": 1, "unit": "%"},
        "versioning": {"type": "", "params": {}, "cleanupIntervalS": 3600},
        "copiers": 0,
        "pullerMaxPendingKiB": 0,
        "hashers": 0,
        "order": "random",
        "pullerPauseS": 0,
        "maxConflicts": 10,
        "disableSparseFiles": False,
        "disableTempIndexes": False,
        "paused": False,
        "weakHashThresholdPct": 25,
        "markerName": ".stfolder",
        "copyOwnershipFromParent": False,
        "modTimeWindowS": 0,
        "maxConcurrentWrites": 2,
        "disableFsync": False,
        "blockPullOrder": "standard",
        "copyRangeMethod": "standard",
        "caseSensitiveFS": False,
        "junctionsAsDirs": False,
        "syncOwnership": False,
        "sendOwnership": False,
        "syncXattrs": False,
        "sendXattrs": False,
        "xattrFilter": {"entries": None, "regexes": None}
    })
    print("   ✓ Added Niodoo-Final folder config")

# Update RunPod config with folders
if not models_folder_exists or not niodoo_folder_exists:
    config_json = json.dumps(runpod_config)
    ssh_cmd = f'''ssh -T root@38.80.152.72 -p 30572 -i ~/.ssh/id_ed25519 "curl -s -X PUT -H 'X-API-Key: {RUNPOD_API_KEY}' -H 'Content-Type: application/json' -d '{config_json}' http://localhost:8384/rest/system/config"'''
    result = subprocess.check_output(ssh_cmd, shell=True, text=True)
    print("   ✓ Updated RunPod config with folders")

# Step 4: Accept folders on local side
print("\n4. Accepting folders on local machine...")
config = requests.get(f"{LOCAL_URL}/rest/system/config", headers=headers).json()

# Accept models folder
models_exists = any(f['id'] == 'runpod-models' for f in config.get('folders', []))
if not models_exists:
    config['folders'].append({
        "id": "runpod-models",
        "label": "RunPod Models",
        "filesystemType": "basic",
        "path": "/home/beelink/niodoo-tcs/models",
        "type": "sendreceive",
        "devices": [{"deviceID": RUNPOD_DEVICE_ID, "introducedBy": "", "encryptionPassword": ""}],
        "rescanIntervalS": 3600,
        "fsWatcherEnabled": True,
        "fsWatcherDelayS": 10,
        "ignorePerms": False,
        "autoNormalize": True,
        "minDiskFree": {"value": 1, "unit": "%"},
        "versioning": {"type": "", "params": {}, "cleanupIntervalS": 3600},
        "copiers": 0,
        "pullerMaxPendingKiB": 0,
        "hashers": 0,
        "order": "random",
        "pullerPauseS": 0,
        "maxConflicts": 10,
        "disableSparseFiles": False,
        "disableTempIndexes": False,
        "paused": False,
        "weakHashThresholdPct": 25,
        "markerName": ".stfolder",
        "copyOwnershipFromParent": False,
        "modTimeWindowS": 0,
        "maxConcurrentWrites": 2,
        "disableFsync": False,
        "blockPullOrder": "standard",
        "copyRangeMethod": "standard",
        "caseSensitiveFS": False,
        "junctionsAsDirs": False,
        "syncOwnership": False,
        "sendOwnership": False,
        "syncXattrs": False,
        "sendXattrs": False,
        "xattrFilter": {"entries": None, "regexes": None}
    })
    print("   ✓ Added models folder to local")

# Accept Niodoo-Final folder
niodoo_exists = any(f['id'] == 'niodoo-final' for f in config.get('folders', []))
if not niodoo_exists:
    config['folders'].append({
        "id": "niodoo-final",
        "label": "Niodoo-Final",
        "filesystemType": "basic",
        "path": "/home/beelink/niodoo-tcs/Niodoo-Final",
        "type": "sendreceive",
        "devices": [{"deviceID": RUNPOD_DEVICE_ID, "introducedBy": "", "encryptionPassword": ""}],
        "rescanIntervalS": 3600,
        "fsWatcherEnabled": True,
        "fsWatcherDelayS": 10,
        "ignorePerms": False,
        "autoNormalize": True,
        "minDiskFree": {"value": 1, "unit": "%"},
        "versioning": {"type": "", "params": {}, "cleanupIntervalS": 3600},
        "copiers": 0,
        "pullerMaxPendingKiB": 0,
        "hashers": 0,
        "order": "random",
        "pullerPauseS": 0,
        "maxConflicts": 10,
        "disableSparseFiles": False,
        "disableTempIndexes": False,
        "paused": False,
        "weakHashThresholdPct": 25,
        "markerName": ".stfolder",
        "copyOwnershipFromParent": False,
        "modTimeWindowS": 0,
        "maxConcurrentWrites": 2,
        "disableFsync": False,
        "blockPullOrder": "standard",
        "copyRangeMethod": "standard",
        "caseSensitiveFS": False,
        "junctionsAsDirs": False,
        "syncOwnership": False,
        "sendOwnership": False,
        "syncXattrs": False,
        "sendXattrs": False,
        "xattrFilter": {"entries": None, "regexes": None}
    })
    print("   ✓ Added Niodoo-Final folder to local")

if not models_exists or not niodoo_exists:
    resp = requests.put(f"{LOCAL_URL}/rest/system/config", headers=headers, json=config)
    print(f"   ✓ Updated local config: {resp.status_code}")

print("\n=== Setup Complete! ===")
print("Folders should now be syncing. Check Syncthing UI to verify.")


