#!/bin/bash
# Complete demo script - runs everything needed to show live learning

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     🚀 NIODOO TCS - LIVE LEARNING DEMO 🚀                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $METRICS_PID 2>/dev/null || true
    docker-compose -f docker-compose.monitoring.yml down 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Step 1: Start monitoring stack
echo -e "${BLUE}📊 Starting monitoring stack (Prometheus + Grafana)...${NC}"
docker-compose -f docker-compose.monitoring.yml up -d
sleep 5

# Step 2: Start metrics server
echo -e "${BLUE}🔧 Starting metrics server on port 9091...${NC}"
cd niodoo_real_integrated
cargo build --bin metrics_server --release &
METRICS_PID=$!
sleep 3

# Check if metrics server is running
if curl -s http://localhost:9091/health > /dev/null; then
    echo -e "${GREEN}✓ Metrics server running${NC}"
else
    echo -e "${RED}✗ Metrics server failed to start${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Your Demo is Ready!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📊 Dashboard:${NC} http://localhost:3000"
echo -e "${YELLOW}🔧 Metrics:${NC} http://localhost:9091/metrics"
echo -e "${YELLOW}🏥 Health:${NC} http://localhost:9091/health"
echo ""
echo -e "${CYAN}Login:${NC} admin / niodoo123"
echo ""
echo -e "${BLUE}📝 Now run prompts to generate learning data:${NC}"
echo ""
echo -e "${GREEN}# Run prompts (will show learning metrics)${NC}"
echo "cargo run --release -- -p \"Explain quantum entanglement\" -n 10"
echo ""
echo -e "${GREEN}# Watch learning happen in Grafana:${NC}"
echo "Open http://localhost:3000 and watch entropy decrease"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop everything${NC}"
echo ""

# Wait and monitor
while true; do
    sleep 10
    # Keep running until Ctrl+C
done

