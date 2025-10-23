# ASUS ROG Zephyrus G14 Audio Fix

## 🎵 PROBLEM SOLVED
Your laptop has **separate amplifier controls** (AMP1 & AMP2) that weren't syncing with the main volume controls (Master/Speaker). This caused:
- Either **HELL LOUD** (amps at 94%) or **OFF**
- Adjusting volume in KDE only changed tweeters, not amplifiers
- **Shitty sound** when amps/tweets weren't balanced

## ✅ SOLUTION IMPLEMENTED
Created an auto-sync service that **dynamically adjusts AMP1 and AMP2** whenever you change Master volume.

### How It Works:
- **Master Volume: 0%** → AMPs at 50% (224/448)
- **Master Volume: 50%** → AMPs at 75% (336/448)  
- **Master Volume: 100%** → AMPs at 100% (448/448)

This gives you **full dynamic range** without ever sounding like shit.

## 🔧 FILES CREATED
1. **`sync-audio-amps.sh`** - The monitoring script
2. **`~/.config/systemd/user/audio-amp-sync.service`** - Auto-start service

## 📊 CHECKING STATUS
```bash
# Check if service is running
systemctl --user status audio-amp-sync.service

# View live logs
journalctl --user -u audio-amp-sync.service -f

# Manual volume test
amixer -c 3 sset Master 70%
```

## 🎛️ MANUAL CONTROLS (if needed)
```bash
# Check current AMP levels
amixer -c 3 sget "AMP1 Speaker"
amixer -c 3 sget "AMP2 Speaker"

# Manually set AMP levels (0-448)
amixer -c 3 sset "AMP1 Speaker" 336
amixer -c 3 sset "AMP2 Speaker" 336

# Restart the sync service
systemctl --user restart audio-amp-sync.service
```

## 🔄 IF YOU NEED TO DISABLE IT
```bash
systemctl --user stop audio-amp-sync.service
systemctl --user disable audio-amp-sync.service
```

## 🎚️ USING KDE VOLUME CONTROLS
**Just use them normally!** The script monitors Master volume and syncs the AMPs automatically.

- Volume buttons on keyboard: ✅ Work
- Volume slider in system tray: ✅ Work  
- `amixer` commands: ✅ Work
- PulseAudio/PipeWire controls: ✅ Work

## 📝 TECHNICAL DETAILS
- **Audio Chip**: Realtek ALC285 (0x10ec0285)
- **Subsystem ID**: 0x10431024
- **Card**: hw:3 (Family 17h/19h/1ah HD Audio Controller)
- **Amplifiers**: CS35L56 (independent from main codec)
- **Monitor Rate**: 5 checks/second (0.2s delay)

## 🆙 SYSTEM UPDATES COMPLETED
- **BIOS**: GA403WW.307 (07/09/2025) - ✅ Latest
- **Linux Kernel**: 6.14.0-33-generic - ✅ Up to date
- **Firmware**: All devices updated via fwupdmgr
- **Packages**: System fully upgraded

---

**Service starts automatically on login. Enjoy your properly balanced audio! 🔊**