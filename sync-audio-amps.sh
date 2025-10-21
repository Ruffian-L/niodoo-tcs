#!/bin/bash
# ASUS ROG Zephyrus G14 Audio AMP Sync Script
# Keeps AMP1 and AMP2 Speaker volumes synced with Master volume changes

CARD="3"

# Function to calculate AMP volume based on Master percentage
calc_amp_volume() {
    local master_pct=$1
    # AMPs range: 0-448
    # Scale AMPs from 224 (50%) to 448 (100%) based on Master 0-100%
    echo $(( 224 + (master_pct * 224 / 100) ))
}

# Get current Master volume percentage
get_master_volume() {
    amixer -c "$CARD" sget Master | grep -oP '\[\K[0-9]+(?=%\])' | head -1
}

# Set AMP volumes
set_amp_volumes() {
    local amp_vol=$1
    amixer -c "$CARD" sset "AMP1 Speaker" "$amp_vol" > /dev/null 2>&1
    amixer -c "$CARD" sset "AMP2 Speaker" "$amp_vol" > /dev/null 2>&1
}

# Monitor for volume changes
prev_master_vol=$(get_master_volume)

echo "🔊 Audio AMP Sync started - Monitoring Master volume on card $CARD"
echo "   AMPs will scale from 50-100% based on Master 0-100%"

while true; do
    current_master_vol=$(get_master_volume)
    
    if [ "$current_master_vol" != "$prev_master_vol" ]; then
        amp_volume=$(calc_amp_volume "$current_master_vol")
        set_amp_volumes "$amp_volume"
        echo "[$(date '+%H:%M:%S')] Master: ${current_master_vol}% -> AMP: ${amp_volume}/448 ($(( amp_volume * 100 / 448 ))%)"
        prev_master_vol=$current_master_vol
    fi
    
    sleep 0.2  # Check 5 times per second
done