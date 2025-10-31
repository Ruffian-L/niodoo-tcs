#!/bin/bash
# Easy dashboard launcher script with terminal safety
# Usage: ./dashboard.sh [log_file]

LOG_FILE="${1:-/tmp/soak_10k_cycles.log}"

# Terminal cleanup function
cleanup() {
    # Restore terminal to sane state
    stty sane
    tput cnorm  # Show cursor
    tput rmcup  # Exit alternate screen
    echo -e "\n\n✨ Terminal restored!"
}

# Set trap to cleanup on exit/error
trap cleanup EXIT INT TERM

# Ensure we have a proper terminal
export TERM=${TERM:-xterm-256color}

echo "🚀 Starting Niodoo Dashboard..."
echo "📋 Monitoring: $LOG_FILE"
echo ""
echo "Press 'q' to quit"
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Error: Log file '$LOG_FILE' not found!"
    echo ""
    echo "Available log files:"
    ls -lh /tmp/soak*.log 2>/dev/null || echo "  No soak log files found"
    exit 1
fi

# Run pre-built binary directly
/workspace/Niodoo-Final/target/release/dashboard "$LOG_FILE" 2>&1

# Cleanup will run automatically due to trap



