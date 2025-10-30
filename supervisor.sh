#!/bin/bash
# Compatibility wrapper around unified_service_manager.sh

set -euo pipefail

ROOT="/workspace/Niodoo-Final"
MANAGER="$ROOT/unified_service_manager.sh"

if [[ ! -x "$MANAGER" ]]; then
    echo "Error: $MANAGER not found or not executable" >&2
    exit 1
fi

usage() {
    cat <<'EOF'
Usage: supervisor.sh {start|stop|restart|status|health|monitor|start-service|stop-service|restart-service} [...]

Commands delegate to unified_service_manager.sh for lifecycle management.
EOF
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
    usage
    exit 1
fi

shift || true

case "$cmd" in
    start|stop|restart|status|health|monitor)
        exec "$MANAGER" "$cmd" "$@"
        ;;
    start-service|stop-service|restart-service)
        exec "$MANAGER" "$cmd" "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac

