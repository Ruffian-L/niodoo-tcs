#!/bin/bash
# Start all services in background with auto-restart
# Saves to /workspace network drive

cd /workspace/Niodoo-Final

# Start supervisor monitor in background
nohup ./supervisor.sh monitor > /tmp/supervisor.log 2>&1 &

# Wait a moment for it to start
sleep 2

# Start all services
./supervisor.sh start

echo "✅ Services started. Supervisor monitoring in background."
echo "📊 Check status: ./supervisor.sh status"
echo "📝 Supervisor log: tail -f /tmp/supervisor.log"

