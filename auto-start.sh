#!/bin/bash
# Auto-start services on pod boot
# This script checks and starts all services if they're not running

echo "🚀 Auto-starting services on pod boot..."

# Wait for network to be ready
sleep 3

# Start all services using supervisor
ROOT=${NIODOO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$ROOT"
./supervisor.sh start

# Start supervisor monitor in background
nohup ./supervisor.sh monitor > /tmp/supervisor.log 2>&1 &

echo "✅ Auto-start complete. Services running in background."
echo "📊 Check status: cd \"$ROOT\" && ./supervisor.sh status"

