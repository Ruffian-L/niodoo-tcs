#!/bin/bash

# Device IDs
LAPTOP_ID="PCDHNJB-C26NMKS-JF4TISV-C5NECQV-E24JXNG-VOITHMM-YV4SBEK-ODYURAC"
BEELINK_ID="74RY2P2-DNQB7KW-3BS55CN-424VD7C-F62EDYG-HCHDU26-AUJ3NP7-PEV7TA2"

# Tailscale IPs
LAPTOP_IP="100.126.84.41"
BEELINK_IP="100.113.10.90"

# Folder paths
LAPTOP_FOLDER="/home/ruffian/Desktop/Niodoo-Final"
BEELINK_FOLDER="/home/beelink/Niodoo-Final"

echo "Setting up Syncthing two-way sync for Niodoo-Final..."

# Create folder on beelink if it doesn't exist
ssh -i ~/.ssh/temp_beelink_key beelink@$BEELINK_IP "mkdir -p $BEELINK_FOLDER"

# Generate unique folder ID
FOLDER_ID="niodoo-final-$(date +%s)"

# Configure laptop (using syncthing cli config)
echo "Configuring laptop..."

# Add remote device (beelink)
syncthing cli config devices add --device-id "$BEELINK_ID" || true

# Add shared folder
syncthing cli config folders add --id "$FOLDER_ID" --label "Niodoo-Final" --path "$LAPTOP_FOLDER" || true

# Share folder with beelink
syncthing cli config folders "$FOLDER_ID" devices add --device-id "$BEELINK_ID" || true

# Configure beelink
echo "Configuring beelink..."
ssh -i ~/.ssh/temp_beelink_key beelink@$BEELINK_IP "
    # Add remote device (laptop)
    syncthing cli config devices add --device-id '$LAPTOP_ID' || true

    # Add shared folder
    syncthing cli config folders add --id '$FOLDER_ID' --label 'Niodoo-Final' --path '$BEELINK_FOLDER' || true

    # Share folder with laptop
    syncthing cli config folders '$FOLDER_ID' devices add --device-id '$LAPTOP_ID' || true
"

# Restart services to apply changes
echo "Restarting Syncthing services..."
systemctl --user restart syncthing.service
ssh -i ~/.ssh/temp_beelink_key beelink@$BEELINK_IP "systemctl --user restart syncthing.service"

echo "Done! Two-way sync configured."
echo ""
echo "Web interfaces:"
echo "  Laptop:  http://localhost:8384"
echo "  Beelink: http://$BEELINK_IP:8384"
echo ""
echo "You can access these to monitor sync status and make additional configurations."
echo "The folder will sync automatically between both machines."