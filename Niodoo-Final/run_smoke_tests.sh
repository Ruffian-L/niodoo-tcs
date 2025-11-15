#!/bin/bash
# Individual smoke tests for each endpoint

echo "=== SMOKE TESTING ALL ENDPOINTS INDIVIDUALLY ===" > /tmp/smoke_results.txt
echo "" >> /tmp/smoke_results.txt

echo "1. llama.cpp Server (Port 8000):" >> /tmp/smoke_results.txt
curl -s --max-time 5 http://127.0.0.1:8000/health >> /tmp/smoke_results.txt 2>&1 && echo " ✅ /health: ONLINE" >> /tmp/smoke_results.txt || echo " ❌ /health: OFFLINE" >> /tmp/smoke_results.txt

curl -s --max-time 5 http://127.0.0.1:8000/v1/models | python3 -c "import sys, json; d=json.load(sys.stdin); print(' ✅ /v1/models: ONLINE -', len(d.get('data', [])), 'models')" >> /tmp/smoke_results.txt 2>&1 || echo " ❌ /v1/models: OFFLINE" >> /tmp/smoke_results.txt

echo "" >> /tmp/smoke_results.txt
echo "2. Training Service (Port 8002):" >> /tmp/smoke_results.txt
curl -s --max-time 5 http://127.0.0.1:8002/health | python3 -c "import sys, json; d=json.load(sys.stdin); print(' ✅ /health: ONLINE -', d.get('status', 'unknown'))" >> /tmp/smoke_results.txt 2>&1 || echo " ❌ /health: OFFLINE" >> /tmp/smoke_results.txt

curl -s --max-time 5 http://127.0.0.1:8002/metrics | head -1 >> /tmp/smoke_results.txt 2>&1 && echo " ✅ /metrics: ONLINE" >> /tmp/smoke_results.txt || echo " ❌ /metrics: OFFLINE" >> /tmp/smoke_results.txt

curl -s --max-time 5 http://127.0.0.1:8002/training/jobs | python3 -c "import sys, json; d=json.load(sys.stdin); print(' ✅ /training/jobs: ONLINE')" >> /tmp/smoke_results.txt 2>&1 || echo " ❌ /training/jobs: OFFLINE" >> /tmp/smoke_results.txt

curl -s --max-time 5 http://127.0.0.1:8002/training/adapters | python3 -c "import sys, json; d=json.load(sys.stdin); print(' ✅ /training/adapters: ONLINE')" >> /tmp/smoke_results.txt 2>&1 || echo " ❌ /training/adapters: OFFLINE" >> /tmp/smoke_results.txt

echo "" >> /tmp/smoke_results.txt
echo "3. Ollama Curator (Port 11434):" >> /tmp/smoke_results.txt
curl -s --max-time 5 http://127.0.0.1:11434/api/tags | python3 -c "import sys, json; d=json.load(sys.stdin); models=[m.get('name') for m in d.get('models', [])]; print(' ✅ /api/tags: ONLINE -', len(models), 'models')" >> /tmp/smoke_results.txt 2>&1 || echo " ❌ /api/tags: OFFLINE" >> /tmp/smoke_results.txt

echo "" >> /tmp/smoke_results.txt
echo "4. Qdrant Cloud:" >> /tmp/smoke_results.txt
cd /workspace/Niodoo-Final
QDRANT_API_KEY=$(grep QDRANT_API_KEY .env 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | head -1)
QDRANT_URL="https://068d2af6-e623-468d-bb4e-05dfdc33efae.us-east4-0.gcp.cloud.qdrant.io:6333"
curl -s --max-time 10 -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/health" >> /tmp/smoke_results.txt 2>&1 && echo " ✅ /health: ONLINE" >> /tmp/smoke_results.txt || echo " ❌ /health: OFFLINE" >> /tmp/smoke_results.txt

curl -s --max-time 10 -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections" | python3 -c "import sys, json; d=json.load(sys.stdin); cols=[c['name'] for c in d.get('result', {}).get('collections', [])]; print(' ✅ /collections: ONLINE -', len(cols), 'collections')" >> /tmp/smoke_results.txt 2>&1 || echo " ❌ /collections: OFFLINE" >> /tmp/smoke_results.txt

echo "" >> /tmp/smoke_results.txt
echo "5. Visualization Bridge:" >> /tmp/smoke_results.txt
curl -s --max-time 5 http://127.0.0.1:8080/ | head -1 >> /tmp/smoke_results.txt 2>&1 && echo " ✅ HTTP (8080): ONLINE" >> /tmp/smoke_results.txt || echo " ❌ HTTP (8080): OFFLINE" >> /tmp/smoke_results.txt

timeout 2 bash -c 'echo > /dev/tcp/127.0.0.1/8765' 2>&1 && echo " ✅ WebSocket (8765): OPEN" >> /tmp/smoke_results.txt || echo " ❌ WebSocket (8765): CLOSED" >> /tmp/smoke_results.txt

echo "" >> /tmp/smoke_results.txt
echo "6. Telemetry Server (Port 9999):" >> /tmp/smoke_results.txt
timeout 2 bash -c 'echo > /dev/tcp/127.0.0.1/9999' 2>&1 && echo " ✅ OPEN" >> /tmp/smoke_results.txt || echo " ⚠️  CLOSED (expected - starts with test)" >> /tmp/smoke_results.txt

cat /tmp/smoke_results.txt

