#!/usr/bin/env python3
"""Smoke test all endpoints from HOWTOLATEST.md"""
import urllib.request
import json
import socket
import os
import sys

def test_http(url, timeout=5, headers=None):
    try:
        req = urllib.request.Request(url)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode()
    except Exception as e:
        return None, str(e)

def test_tcp(host, port, timeout=2):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

print('=' * 70)
print('SMOKE TESTING ALL ENDPOINTS FROM HOWTOLATEST.md')
print('=' * 70)
print()

results = []

# 1. llama.cpp Server
print('1. llama.cpp Server (Port 8000):')
code, resp = test_http('http://127.0.0.1:8000/health')
if code == 200:
    print('   ✅ /health: ONLINE')
    results.append(('llama.cpp /health', True))
else:
    print(f'   ❌ /health: OFFLINE ({resp[:50]})')
    results.append(('llama.cpp /health', False))

code, resp = test_http('http://127.0.0.1:8000/v1/models')
if code == 200:
    try:
        data = json.loads(resp)
        models = [m.get('id') for m in data.get('data', [])]
        print(f'   ✅ /v1/models: ONLINE ({len(models)} models)')
        results.append(('llama.cpp /v1/models', True))
    except:
        print('   ✅ /v1/models: ONLINE')
        results.append(('llama.cpp /v1/models', True))
else:
    print(f'   ❌ /v1/models: OFFLINE')
    results.append(('llama.cpp /v1/models', False))

# 2. Training Service
print()
print('2. Training Service (Port 8002):')
code, resp = test_http('http://127.0.0.1:8002/health')
if code == 200:
    print('   ✅ /health: ONLINE')
    results.append(('training /health', True))
else:
    print(f'   ❌ /health: OFFLINE')
    results.append(('training /health', False))

code, resp = test_http('http://127.0.0.1:8002/metrics')
if code == 200:
    print('   ✅ /metrics: ONLINE')
    results.append(('training /metrics', True))
else:
    print(f'   ❌ /metrics: OFFLINE')
    results.append(('training /metrics', False))

code, resp = test_http('http://127.0.0.1:8002/training/jobs')
if code == 200:
    print('   ✅ /training/jobs: ONLINE')
    results.append(('training /jobs', True))
else:
    print(f'   ❌ /training/jobs: OFFLINE')
    results.append(('training /jobs', False))

code, resp = test_http('http://127.0.0.1:8002/training/adapters')
if code == 200:
    print('   ✅ /training/adapters: ONLINE')
    results.append(('training /adapters', True))
else:
    print(f'   ❌ /training/adapters: OFFLINE')
    results.append(('training /adapters', False))

# 3. Ollama
print()
print('3. Ollama Curator (Port 11434):')
code, resp = test_http('http://127.0.0.1:11434/api/tags')
if code == 200:
    try:
        data = json.loads(resp)
        models = [m.get('name') for m in data.get('models', [])]
        print(f'   ✅ /api/tags: ONLINE ({len(models)} models)')
        results.append(('ollama /api/tags', True))
    except:
        print('   ✅ /api/tags: ONLINE')
        results.append(('ollama /api/tags', True))
else:
    print(f'   ❌ /api/tags: OFFLINE')
    results.append(('ollama /api/tags', False))

# 4. Qdrant Cloud
print()
print('4. Qdrant Cloud:')
api_key = None
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if 'QDRANT_API_KEY' in line and '=' in line:
                api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                break

qdrant_url = 'https://068d2af6-e623-468d-bb4e-05dfdc33efae.us-east4-0.gcp.cloud.qdrant.io:6333'
if api_key:
    code, resp = test_http(f'{qdrant_url}/health', headers={'api-key': api_key})
    if code == 200:
        print('   ✅ /health: ONLINE')
        results.append(('qdrant /health', True))
    else:
        print(f'   ❌ /health: OFFLINE')
        results.append(('qdrant /health', False))
    
    code, resp = test_http(f'{qdrant_url}/collections', headers={'api-key': api_key})
    if code == 200:
        try:
            data = json.loads(resp)
            cols = [c['name'] for c in data.get('result', {}).get('collections', [])]
            print(f'   ✅ /collections: ONLINE ({len(cols)} collections: {", ".join(cols)})')
            results.append(('qdrant /collections', True))
        except:
            print('   ✅ /collections: ONLINE')
            results.append(('qdrant /collections', True))
else:
    print('   ⚠️  API key not found')
    results.append(('qdrant', False))

# 5. Visualization Bridge
print()
print('5. Visualization Bridge:')
code, resp = test_http('http://127.0.0.1:8080/')
if code == 200:
    print('   ✅ HTTP (8080): ONLINE')
    results.append(('bridge HTTP', True))
else:
    print(f'   ❌ HTTP (8080): OFFLINE')
    results.append(('bridge HTTP', False))

ws_open = test_tcp('127.0.0.1', 8765)
if ws_open:
    print('   ✅ WebSocket (8765): OPEN')
    results.append(('bridge WebSocket', True))
else:
    print('   ❌ WebSocket (8765): CLOSED')
    results.append(('bridge WebSocket', False))

# 6. Telemetry
print()
print('6. Telemetry Server (Port 9999):')
tel_open = test_tcp('127.0.0.1', 9999)
if tel_open:
    print('   ✅ OPEN')
    results.append(('telemetry', True))
else:
    print('   ⚠️  CLOSED (starts with test when NIODOO_TELEMETRY_ENABLED=true)')
    results.append(('telemetry', False))

# Summary
print()
print('=' * 70)
print('SMOKE TEST SUMMARY')
print('=' * 70)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f'Passed: {passed}/{total}')
print()
for name, ok in results:
    status = '✅' if ok else '❌'
    print(f'{status} {name}')
print('=' * 70)
sys.exit(0 if passed == total else 1)

