#!/bin/bash
# Wipe Qdrant Collections Script
# This script deletes all Qdrant collections to clear corruption and start fresh
# Based on architectural remediation for mutex poisoning cascade

set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-niodoo_memories}"

echo "🔧 Wiping Qdrant collections to clear corruption..."
echo "Qdrant URL: $QDRANT_URL"
echo ""

# List all collections
echo "📋 Listing existing collections..."
COLLECTIONS=$(curl -s "${QDRANT_URL}/collections" | jq -r '.result.collections[].name' 2>/dev/null || echo "")

if [ -z "$COLLECTIONS" ]; then
    echo "✅ No collections found. Qdrant is already clean."
    exit 0
fi

echo "Found collections:"
echo "$COLLECTIONS" | while read -r collection; do
    if [ -n "$collection" ]; then
        echo "  - $collection"
    fi
done

echo ""
read -p "⚠️  This will DELETE ALL collections. Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Aborted."
    exit 1
fi

# Delete each collection
echo ""
echo "🗑️  Deleting collections..."
echo "$COLLECTIONS" | while read -r collection; do
    if [ -n "$collection" ]; then
        echo "  Deleting: $collection"
        curl -X DELETE "${QDRANT_URL}/collections/${collection}" 2>/dev/null || true
        echo "    ✅ Deleted $collection"
    fi
done

echo ""
echo "✅ All collections wiped. Collections will be recreated on next pipeline startup."
echo ""
echo "📝 Next steps:"
echo "   1. Restart the pipeline to recreate collections"
echo "   2. The new Actor Model pattern will prevent mutex poisoning"
echo "   3. Write batching will reduce Qdrant contention"



