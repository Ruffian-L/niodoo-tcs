#!/usr/bin/env python3
import json
import requests
import subprocess
import time

# Device IDs and IPs
LAPTOP_ID = "PCDHNJB-C26NMKS-JF4TISV-C5NECQV-E24JXNG-VOITHMM-YV4SBEK-ODYURAC"
BEELINK_ID = "74RY2P2-DNQB7KW-3BS55CN-424VD7C-F62EDYG-HCHDU26-AUJ3NP7-PEV7TA2"
LAPTOP_IP = "100.126.84.41"
BEELINK_IP = "100.113.10.90"

# Folder settings
FOLDER_ID = "niodoo-final"
LAPTOP_FOLDER = "/home/ruffian/Desktop/Niodoo-Final"
BEELINK_FOLDER = "/home/beelink/Niodoo-Final"

def get_config(host="localhost"):
    """Get current Syncthing configuration"""
    url = f"http://{host}:8384/rest/system/config"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def save_config(config, host="localhost"):
    """Save updated Syncthing configuration"""
    url = f"http://{host}:8384/rest/system/config"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(config))
    return response.status_code == 200

def restart_syncthing(host="localhost"):
    """Restart Syncthing to apply changes"""
    url = f"http://{host}:8384/rest/system/restart"
    response = requests.post(url)
    return response.status_code == 200

def configure_laptop():
    """Configure laptop's Syncthing"""
    print("Configuring laptop...")
    config = get_config()

    if not config:
        print("Failed to get laptop config")
        return False

    # Add beelink as a device
    device_exists = False
    for device in config.get("devices", []):
        if device["deviceID"] == BEELINK_ID:
            device_exists = True
            break

    if not device_exists:
        config["devices"].append({
            "deviceID": BEELINK_ID,
            "name": "Beelink",
            "addresses": [f"tcp://{BEELINK_IP}:22000"],
            "compression": "metadata",
            "certName": "",
            "introducer": False,
            "skipIntroductionRemovals": False,
            "introducedBy": "",
            "paused": False,
            "allowedNetworks": [],
            "autoAcceptFolders": False,
            "maxSendKbps": 0,
            "maxRecvKbps": 0,
            "ignoredFolders": [],
            "pendingFolders": [],
            "maxRequestKiB": 0
        })

    # Add or update folder
    folder_exists = False
    for folder in config.get("folders", []):
        if folder["id"] == FOLDER_ID:
            folder_exists = True
            # Ensure beelink is in the device list
            device_in_folder = False
            for device in folder.get("devices", []):
                if device["deviceID"] == BEELINK_ID:
                    device_in_folder = True
                    break
            if not device_in_folder:
                folder["devices"].append({
                    "deviceID": BEELINK_ID,
                    "introducedBy": "",
                    "encryptionPassword": ""
                })
            break

    if not folder_exists:
        config["folders"].append({
            "id": FOLDER_ID,
            "label": "Niodoo-Final",
            "filesystemType": "basic",
            "path": LAPTOP_FOLDER,
            "type": "sendreceive",
            "devices": [
                {
                    "deviceID": LAPTOP_ID,
                    "introducedBy": "",
                    "encryptionPassword": ""
                },
                {
                    "deviceID": BEELINK_ID,
                    "introducedBy": "",
                    "encryptionPassword": ""
                }
            ],
            "rescanIntervalS": 3600,
            "fsWatcherEnabled": True,
            "fsWatcherDelayS": 10,
            "ignorePerms": False,
            "autoNormalize": True,
            "minDiskFree": {"value": 1, "unit": "%"},
            "versioning": {"type": "", "params": {}},
            "copiers": 0,
            "pullerMaxPendingKiB": 0,
            "hashers": 0,
            "order": "random",
            "ignoreDelete": False,
            "scanProgressIntervalS": 0,
            "pullerPauseS": 0,
            "maxConflicts": 10,
            "disableSparseFiles": False,
            "disableTempIndexes": False,
            "paused": False,
            "weakHashThresholdPct": 25,
            "markerName": ".stfolder",
            "copyOwnershipFromParent": False,
            "modTimeWindowS": 0
        })

    if save_config(config):
        print("Laptop configuration saved")
        return True
    else:
        print("Failed to save laptop configuration")
        return False

def configure_beelink():
    """Configure beelink's Syncthing via SSH tunnel"""
    print("Configuring beelink...")

    # Create SSH tunnel for beelink's Syncthing
    tunnel_cmd = f"ssh -i ~/.ssh/temp_beelink_key -L 18384:{BEELINK_IP}:8384 beelink@{BEELINK_IP} -N"
    tunnel_proc = subprocess.Popen(tunnel_cmd, shell=True)

    try:
        time.sleep(2)  # Wait for tunnel to establish

        config = get_config("localhost:18384")
        if not config:
            print("Failed to get beelink config")
            return False

        # Add laptop as a device
        device_exists = False
        for device in config.get("devices", []):
            if device["deviceID"] == LAPTOP_ID:
                device_exists = True
                break

        if not device_exists:
            config["devices"].append({
                "deviceID": LAPTOP_ID,
                "name": "Laptop",
                "addresses": [f"tcp://{LAPTOP_IP}:22000"],
                "compression": "metadata",
                "certName": "",
                "introducer": False,
                "skipIntroductionRemovals": False,
                "introducedBy": "",
                "paused": False,
                "allowedNetworks": [],
                "autoAcceptFolders": False,
                "maxSendKbps": 0,
                "maxRecvKbps": 0,
                "ignoredFolders": [],
                "pendingFolders": [],
                "maxRequestKiB": 0
            })

        # Add or update folder
        folder_exists = False
        for folder in config.get("folders", []):
            if folder["id"] == FOLDER_ID:
                folder_exists = True
                # Ensure laptop is in the device list
                device_in_folder = False
                for device in folder.get("devices", []):
                    if device["deviceID"] == LAPTOP_ID:
                        device_in_folder = True
                        break
                if not device_in_folder:
                    folder["devices"].append({
                        "deviceID": LAPTOP_ID,
                        "introducedBy": "",
                        "encryptionPassword": ""
                    })
                break

        if not folder_exists:
            config["folders"].append({
                "id": FOLDER_ID,
                "label": "Niodoo-Final",
                "filesystemType": "basic",
                "path": BEELINK_FOLDER,
                "type": "sendreceive",
                "devices": [
                    {
                        "deviceID": BEELINK_ID,
                        "introducedBy": "",
                        "encryptionPassword": ""
                    },
                    {
                        "deviceID": LAPTOP_ID,
                        "introducedBy": "",
                        "encryptionPassword": ""
                    }
                ],
                "rescanIntervalS": 3600,
                "fsWatcherEnabled": True,
                "fsWatcherDelayS": 10,
                "ignorePerms": False,
                "autoNormalize": True,
                "minDiskFree": {"value": 1, "unit": "%"},
                "versioning": {"type": "", "params": {}},
                "copiers": 0,
                "pullerMaxPendingKiB": 0,
                "hashers": 0,
                "order": "random",
                "ignoreDelete": False,
                "scanProgressIntervalS": 0,
                "pullerPauseS": 0,
                "maxConflicts": 10,
                "disableSparseFiles": False,
                "disableTempIndexes": False,
                "paused": False,
                "weakHashThresholdPct": 25,
                "markerName": ".stfolder",
                "copyOwnershipFromParent": False,
                "modTimeWindowS": 0
            })

        if save_config(config, "localhost:18384"):
            print("Beelink configuration saved")
            return True
        else:
            print("Failed to save beelink configuration")
            return False

    finally:
        tunnel_proc.terminate()
        tunnel_proc.wait()

def main():
    print("Setting up two-way Syncthing sync for Niodoo-Final...")

    # Create folder on beelink if needed
    subprocess.run(f"ssh -i ~/.ssh/temp_beelink_key beelink@{BEELINK_IP} 'mkdir -p {BEELINK_FOLDER}'", shell=True)

    # Configure both machines
    laptop_success = configure_laptop()
    beelink_success = configure_beelink()

    if laptop_success and beelink_success:
        print("\nRestarting Syncthing services...")
        restart_syncthing()

        # Restart beelink via SSH tunnel
        tunnel_cmd = f"ssh -i ~/.ssh/temp_beelink_key -L 18384:{BEELINK_IP}:8384 beelink@{BEELINK_IP} -N"
        tunnel_proc = subprocess.Popen(tunnel_cmd, shell=True)
        time.sleep(2)
        restart_syncthing("localhost:18384")
        tunnel_proc.terminate()
        tunnel_proc.wait()

        print("\n✅ Two-way sync successfully configured!")
        print("\nWeb interfaces:")
        print(f"  Laptop:  http://localhost:8384")
        print(f"  Beelink: http://{BEELINK_IP}:8384")
        print("\nThe Niodoo-Final folder will now sync automatically between both machines.")
    else:
        print("\n❌ Configuration failed. Please check the error messages above.")

if __name__ == "__main__":
    main()