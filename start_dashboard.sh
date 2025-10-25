#!/bin/bash
# Quick start script for Niodoo Dashboard

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          🚀 NIODOO DASHBOARD QUICK START 🚀                 ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check Docker
echo -e "${BLUE}📦 Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found! Please install Docker first.${NC}"
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Step 2: Start monitoring stack
echo ""
echo -e "${BLUE}🚀 Starting monitoring stack...${NC}"
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 10

# Check if services are running
echo ""
echo -e "${BLUE}🔍 Checking services...${NC}"

# Check Prometheus
if curl -s -o /dev/null -w "%{http_code}" http://localhost:9090 | grep -q "200"; then
    echo -e "${GREEN}✓ Prometheus is running at http://localhost:9090${NC}"
else
    echo -e "${YELLOW}⚠️  Prometheus might still be starting...${NC}"
fi

# Check Grafana
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200\|302"; then
    echo -e "${GREEN}✓ Grafana is running at http://localhost:3000${NC}"
else
    echo -e "${YELLOW}⚠️  Grafana might still be starting...${NC}"
fi

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Dashboard Setup Complete!${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}📊 Access your dashboards:${NC}"
echo -e "   Grafana: ${GREEN}http://localhost:3000${NC}"
echo -e "   Login: ${YELLOW}admin / niodoo123${NC}"
echo -e "   Dashboard: Auto-loaded as '🚀 Niodoo Learning Dashboard'"
echo ""
echo -e "${CYAN}🔬 Run tests with metrics:${NC}"
echo -e "   Simple test: ${GREEN}./run_with_metrics.sh \"Your prompt here\" 5${NC}"
echo -e "   Hard problems: ${GREEN}./run_with_metrics.sh \"\$(head -n 1 hard_problems.txt)\" 10${NC}"
echo ""
echo -e "${CYAN}📺 Console monitoring:${NC}"
echo -e "   Live view: ${GREEN}./monitor_live.sh${NC}"
echo -e "   Watch logs: ${GREEN}watch -n 5 'tail -20 logs/*.log | grep entropy'${NC}"
echo ""
echo -e "${CYAN}🛑 To stop everything:${NC}"
echo -e "   ${YELLOW}docker-compose -f docker-compose.monitoring.yml down${NC}"
echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo -e "   • Look for entropy curves going DOWN = AI is learning"
echo -e "   • ROUGE scores going UP = Better quality solutions"
echo -e "   • Balanced threat/healing pie chart = Healthy emotional state"
echo -e "   • Run same prompt multiple times to see learning improvement"
echo ""
