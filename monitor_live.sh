#!/bin/bash
# Live console monitoring for Niodoo metrics

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           🚀 NIODOO LIVE LEARNING MONITOR 🚀                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to extract metrics from logs
show_metrics() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📊 Latest Metrics ($(date '+%H:%M:%S'))${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Find latest log file
    LATEST_LOG=$(ls -t logs/*gauntlet*.log 2>/dev/null | head -1)
    LATEST_PROM=$(ls -t logs/*.prom 2>/dev/null | head -1)
    
    if [ -f "$LATEST_LOG" ]; then
        # Extract entropy (learning indicator)
        ENTROPY=$(grep -o "entropy: [0-9.]*" "$LATEST_LOG" | tail -1 | cut -d' ' -f2)
        if [ ! -z "$ENTROPY" ]; then
            if (( $(echo "$ENTROPY < 0.5" | bc -l) )); then
                echo -e "🧠 Entropy: ${GREEN}$ENTROPY${NC} bits (Low = Good Learning!)"
            else
                echo -e "🧠 Entropy: ${YELLOW}$ENTROPY${NC} bits (Still Learning...)"
            fi
        fi
        
        # Extract quality score
        QUALITY=$(grep -o "rouge_l: [0-9.]*" "$LATEST_LOG" | tail -1 | cut -d' ' -f2)
        if [ ! -z "$QUALITY" ]; then
            if (( $(echo "$QUALITY > 0.75" | bc -l) )); then
                echo -e "✨ Quality: ${GREEN}$QUALITY${NC} (Excellent Solution!)"
            else
                echo -e "✨ Quality: ${YELLOW}$QUALITY${NC} (Decent Solution)"
            fi
        fi
        
        # Extract latency
        LATENCY=$(grep -o "latency_ms: [0-9.]*" "$LATEST_LOG" | tail -1 | cut -d' ' -f2)
        if [ ! -z "$LATENCY" ]; then
            echo -e "⚡ Speed: ${CYAN}$LATENCY${NC} ms"
        fi
        
        # Count threats vs healings
        THREATS=$(grep -c "threat_detected" "$LATEST_LOG" 2>/dev/null || echo "0")
        HEALINGS=$(grep -c "healing_detected" "$LATEST_LOG" 2>/dev/null || echo "0")
        echo -e "🎭 Balance: ${RED}$THREATS threats${NC} / ${GREEN}$HEALINGS healings${NC}"
        
        # Show breakthrough moments
        BREAKTHROUGHS=$(grep -c "breakthrough" "$LATEST_LOG" 2>/dev/null || echo "0")
        if [ "$BREAKTHROUGHS" -gt 0 ]; then
            echo -e "🎯 Breakthroughs: ${PURPLE}$BREAKTHROUGHS${NC} moments of insight!"
        fi
    else
        echo -e "${RED}No log files found yet. Run a test first!${NC}"
    fi
    
    # Check if Prometheus metrics exist
    if [ -f "$LATEST_PROM" ]; then
        echo ""
        echo -e "${BLUE}📈 Prometheus Metrics Available:${NC}"
        echo -e "   File: ${CYAN}$LATEST_PROM${NC}"
        METRIC_COUNT=$(grep -c "^niodoo_" "$LATEST_PROM" 2>/dev/null || echo "0")
        echo -e "   Metrics: ${GREEN}$METRIC_COUNT${NC} data points"
    fi
}

# Function to show recent activity
show_activity() {
    echo ""
    echo -e "${BLUE}📜 Recent Activity:${NC}"
    if [ -f "logs/rut_gauntlet-*.log" ]; then
        tail -5 logs/rut_gauntlet-*.log 2>/dev/null | while read line; do
            if [[ "$line" == *"entropy"* ]]; then
                echo -e "  ${GREEN}→${NC} $line"
            elif [[ "$line" == *"error"* ]] || [[ "$line" == *"ERROR"* ]]; then
                echo -e "  ${RED}→${NC} $line"
            else
                echo -e "  ${CYAN}→${NC} $line"
            fi
        done
    fi
}

# Main monitoring loop
while true; do
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           🚀 NIODOO LIVE LEARNING MONITOR 🚀                ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    show_metrics
    show_activity
    
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Press Ctrl+C to exit | Refreshing every 5 seconds...${NC}"
    
    sleep 5
done
