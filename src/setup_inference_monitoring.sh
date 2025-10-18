#!/bin/bash
# Setup inference monitoring cron job for Niodoo-Feeling system

echo "Setting up inference monitoring cron job..."

# Make sure we're in the project directory
cd "$(dirname "$0")"

# Install cron if not present (for some systems)
if ! command -v crontab &> /dev/null; then
    echo "Installing cron..."
    sudo apt-get update && sudo apt-get install -y cron
fi

# Start cron service if not running
sudo systemctl start cron 2>/dev/null || true
sudo systemctl enable cron 2>/dev/null || true

# Check if inference audit job already exists
if crontab -l 2>/dev/null | grep -q "audit_inference"; then
    echo "⚠️ Inference audit cron job already exists!"
    echo "Current jobs:"
    crontab -l | grep audit_inference
    echo ""
    echo "To modify: crontab -e"
    exit 1
fi

# Add inference audit job (every 15 minutes as requested)
INFERENCE_AUDIT_CMD="*/15 * * * * cd $(pwd) && cargo run --bin audit_inference > inference_log.txt 2>&1"
(crontab -l 2>/dev/null; echo "$INFERENCE_AUDIT_CMD") | crontab -

# Also add a health check every 30 minutes
HEALTH_CHECK_CMD="*/30 * * * * cd $(pwd) && cargo run --bin real_ai_inference -- --input 'system health check' --config config.toml >> inference_log.txt 2>&1"
(crontab -l 2>/dev/null; echo "$HEALTH_CHECK_CMD") | crontab -

echo "✅ Inference monitoring cron jobs setup complete!"
echo "Jobs added:"
echo "  - Every 15 minutes: Run inference audit (as requested)"
echo "  - Every 30 minutes: Run real AI inference health check"
echo ""
echo "To view all cron jobs: crontab -l"
echo "To remove: crontab -e (delete lines containing 'audit_inference' or 'real_ai_inference')"
echo ""
echo "Log file: $(pwd)/inference_log.txt"
