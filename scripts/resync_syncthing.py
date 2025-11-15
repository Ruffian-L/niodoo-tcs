#!/usr/bin/env python3
"""
Resync Syncthing configuration after RunPod instance change.
Updates device IDs and folder configurations on both local and RunPod instances.
"""

import xml.etree.ElementTree as ET
import subprocess
import sys
import os

# Configuration
LOCAL_CONFIG = "/root/.local/state/syncthing/config.xml"
LOCAL_DEVICE_ID = "6MK4WX6-OCG75KW-RRFKSCO-RIH5IP2-2BIB5VG-3QKIIPK-R3J65DI-B2YKCAM"
OLD_RUNPOD_DEVICE_ID = "WJHFRVE-6FGGCZW-TOJ4Q3N-6WBSJSQ-DTFO6XC-ZXC3DZ6-OBEFSTL-YOMC4Q4"
NEW_RUNPOD_DEVICE_ID = "HFVY4E3-B3VTMAX-PKL7FKW-MWYXT3M-EZJVETT-S3HU3LT-3N5FAGD-DTBMHQQ"
RUNPOD_SSH = "root@38.80.152.72"
RUNPOD_PORT = "30868"
RUNPOD_SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
RUNPOD_CONFIG = "/root/.local/state/syncthing/config.xml"

def run_ssh(cmd):
    """Run command on RunPod via SSH"""
    ssh_cmd = f"ssh -T -p {RUNPOD_PORT} -i {RUNPOD_SSH_KEY} {RUNPOD_SSH} '{cmd}'"
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def update_local_config():
    """Update local Syncthing config to use new RunPod device ID"""
    print("=== Updating Local Syncthing Config ===")
    
    # Read config
    tree = ET.parse(LOCAL_CONFIG)
    root = tree.getroot()
    
    # Find and replace old device ID with new one
    devices_updated = 0
    folders_updated = 0
    
    # Update device entries
    for device in root.findall(".//device"):
        device_id = device.get("id")
        if device_id == OLD_RUNPOD_DEVICE_ID:
            device.set("id", NEW_RUNPOD_DEVICE_ID)
            devices_updated += 1
            print(f"  ✓ Updated device entry: {OLD_RUNPOD_DEVICE_ID} → {NEW_RUNPOD_DEVICE_ID}")
    
    # Update folder device references
    for folder in root.findall(".//folder"):
        for device in folder.findall("device"):
            device_id = device.get("id")
            if device_id == OLD_RUNPOD_DEVICE_ID:
                device.set("id", NEW_RUNPOD_DEVICE_ID)
                folders_updated += 1
                print(f"  ✓ Updated folder '{folder.get('id')}' device reference")
    
    # Ensure new device exists in devices section
    devices_section = root.find(".//devices")
    if devices_section is not None:
        device_exists = any(d.get("id") == NEW_RUNPOD_DEVICE_ID for d in devices_section.findall("device"))
        if not device_exists:
            new_device = ET.SubElement(devices_section, "device")
            new_device.set("id", NEW_RUNPOD_DEVICE_ID)
            ET.SubElement(new_device, "encryptionPassword")
            print(f"  ✓ Added new device entry: {NEW_RUNPOD_DEVICE_ID}")
    
    # Write config
    tree.write(LOCAL_CONFIG, encoding="utf-8", xml_declaration=True)
    print(f"✓ Local config updated ({devices_updated} devices, {folders_updated} folders)")
    
    # Restart Syncthing
    print("  Restarting local Syncthing...")
    subprocess.run(["sudo", "systemctl", "restart", "syncthing@root"], check=False)

def update_runpod_config():
    """Update RunPod Syncthing config to add local device and configure folders"""
    print("\n=== Updating RunPod Syncthing Config ===")
    
    # Download config from RunPod
    print("  Downloading RunPod config...")
    stdout, stderr, code = run_ssh(f"cat {RUNPOD_CONFIG}")
    if code != 0:
        print(f"  ✗ Failed to read RunPod config: {stderr}")
        return
    
    # Parse config
    root = ET.fromstring(stdout)
    
    # Add local device to devices section
    devices_section = root.find(".//devices")
    if devices_section is not None:
        device_exists = any(d.get("id") == LOCAL_DEVICE_ID for d in devices_section.findall("device"))
        if not device_exists:
            new_device = ET.SubElement(devices_section, "device")
            new_device.set("id", LOCAL_DEVICE_ID)
            ET.SubElement(new_device, "encryptionPassword")
            print(f"  ✓ Added local device: {LOCAL_DEVICE_ID}")
        else:
            print(f"  ✓ Local device already exists")
    
    # Configure folders
    folders_section = root.find(".//folders")
    if folders_section is None:
        folders_section = ET.SubElement(root.find(".//configuration"), "folders")
    
    # Models folder
    models_folder = None
    for folder in folders_section.findall("folder"):
        if folder.get("id") == "models" or folder.get("path") == "/workspace/models":
            models_folder = folder
            break
    
    if models_folder is None:
        models_folder = ET.SubElement(folders_section, "folder")
        models_folder.set("id", "models")
        models_folder.set("label", "Models")
        models_folder.set("path", "/workspace/models")
        models_folder.set("type", "sendreceive")
        models_folder.set("rescanIntervalS", "3600")
        models_folder.set("fsWatcherEnabled", "true")
        models_folder.set("fsWatcherDelayS", "10")
        models_folder.set("fsWatcherTimeoutS", "0")
        models_folder.set("ignorePerms", "false")
        models_folder.set("autoNormalize", "true")
        ET.SubElement(models_folder, "filesystemType").text = "basic"
        print("  ✓ Created models folder")
    else:
        print("  ✓ Models folder exists")
    
    # Add local device to models folder
    device_exists = any(d.get("id") == LOCAL_DEVICE_ID for d in models_folder.findall("device"))
    if not device_exists:
        device_elem = ET.SubElement(models_folder, "device")
        device_elem.set("id", LOCAL_DEVICE_ID)
        ET.SubElement(device_elem, "encryptionPassword")
        print("  ✓ Added local device to models folder")
    
    # Niodoo-Final folder
    niodoo_folder = None
    for folder in folders_section.findall("folder"):
        if folder.get("id") == "niodoo-final" or folder.get("path") == "/workspace/Niodoo-Final":
            niodoo_folder = folder
            break
    
    if niodoo_folder is None:
        niodoo_folder = ET.SubElement(folders_section, "folder")
        niodoo_folder.set("id", "niodoo-final")
        niodoo_folder.set("label", "Niodoo-Final")
        niodoo_folder.set("path", "/workspace/Niodoo-Final")
        niodoo_folder.set("type", "sendreceive")
        niodoo_folder.set("rescanIntervalS", "3600")
        niodoo_folder.set("fsWatcherEnabled", "true")
        niodoo_folder.set("fsWatcherDelayS", "10")
        niodoo_folder.set("fsWatcherTimeoutS", "0")
        niodoo_folder.set("ignorePerms", "false")
        niodoo_folder.set("autoNormalize", "true")
        ET.SubElement(niodoo_folder, "filesystemType").text = "basic"
        print("  ✓ Created Niodoo-Final folder")
    else:
        print("  ✓ Niodoo-Final folder exists")
    
    # Add local device to Niodoo-Final folder
    device_exists = any(d.get("id") == LOCAL_DEVICE_ID for d in niodoo_folder.findall("device"))
    if not device_exists:
        device_elem = ET.SubElement(niodoo_folder, "device")
        device_elem.set("id", LOCAL_DEVICE_ID)
        ET.SubElement(device_elem, "encryptionPassword")
        print("  ✓ Added local device to Niodoo-Final folder")
    
    # Write config back to RunPod
    xml_str = ET.tostring(root, encoding="unicode")
    # Format XML properly
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(xml_str)
    formatted_xml = dom.toprettyxml(indent="    ")
    
    print("  Uploading updated config to RunPod...")
    # Write to temp file, then copy to RunPod
    with open("/tmp/runpod_syncthing_config.xml", "w") as f:
        f.write(formatted_xml)
    
    # Upload via SSH
    upload_cmd = f"scp -P {RUNPOD_PORT} -i {RUNPOD_SSH_KEY} /tmp/runpod_syncthing_config.xml {RUNPOD_SSH}:{RUNPOD_CONFIG}"
    result = subprocess.run(upload_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ Failed to upload config: {result.stderr}")
        # Try alternative: write directly via SSH
        print("  Trying direct write via SSH...")
        stdout, stderr, code = run_ssh(f"cat > {RUNPOD_CONFIG} << 'EOFXML'\n{formatted_xml}\nEOFXML")
        if code != 0:
            print(f"  ✗ Failed: {stderr}")
            return
    
    print("  ✓ RunPod config updated")
    
    # Restart Syncthing on RunPod
    print("  Restarting RunPod Syncthing...")
    stdout, stderr, code = run_ssh("pkill syncthing; sleep 2; nohup syncthing serve --no-browser --logflags=0 --gui-address=0.0.0.0:8384 > /tmp/syncthing.log 2>&1 &")
    if code == 0:
        print("  ✓ RunPod Syncthing restarted")
    else:
        print(f"  ⚠ Restart may have failed: {stderr}")

def main():
    print("=== Syncthing Resync Script ===")
    print(f"Old RunPod Device: {OLD_RUNPOD_DEVICE_ID}")
    print(f"New RunPod Device: {NEW_RUNPOD_DEVICE_ID}")
    print(f"Local Device: {LOCAL_DEVICE_ID}\n")
    
    if os.geteuid() != 0:
        print("✗ This script must be run as root (for local config access)")
        sys.exit(1)
    
    try:
        update_local_config()
        update_runpod_config()
        print("\n=== Resync Complete ===")
        print("✓ Devices paired")
        print("✓ Folders configured")
        print("\nCheck sync status at: http://localhost:8384/")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

