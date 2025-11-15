#!/bin/bash
echo "🌀 RUNNING CORRECTED MOBIUS VISUALIZATION"
echo "========================================"
echo "Using mathematically correct k-twisted equations from your analysis"
echo ""

if command -v qmlscene >/dev/null 2>&1; then
    echo "🚀 Launching corrected visualization..."
    QT_QPA_PLATFORM=xcb qmlscene viz_standalone_ultimate.qml
else
    echo "❌ qmlscene not found. Install: sudo apt install qtdeclarative5-dev-tools"
    echo "Or run the dashboard: cd dashboard/build && ./niodoo-dashboard"
fi
