#!/bin/bash

# Deploy real NIODOO pipeline to Beelink cluster
# Builds niodoo_real_integrated binary, ensures dependencies, runs smoke tests and Prometheus monitoring

set -e

BEELINK_HOST="beelink"
REMOTE_PATH="/home/beelink/Niodoo-Final"
LOCAL_PATH="."
SWARM_SIZE=${SWARM:-1}
OUTPUT_FORMAT=${OUTPUT:-csv}

echo "Syncing code to Beelink (excluding models and bulky artifacts)..."
rsync -avz \
  --exclude='target/' \
  --exclude='.git/' \
  --exclude='models/' \
  --exclude='data/' \
  --exclude='*.ipynb' \
  "$LOCAL_PATH" "$BEELINK_HOST:$REMOTE_PATH"

echo "Building niodoo_real_integrated on Beelink..."
ssh "$BEELINK_HOST" "cd $REMOTE_PATH && cargo build --release -p niodoo_real_integrated"

echo "Setting up Docker environment..."
ssh "$BEELINK_HOST" "cd $REMOTE_PATH && docker build -t niodoo_real_integrated ."

echo "Starting Prometheus monitoring..."
ssh "$BEELINK_HOST" "cd $REMOTE_PATH && docker rm -f prometheus 2>/dev/null || true && docker run -d -p 9090:9090 --name prometheus prom/prometheus"

echo "Running real-integrated binary on Beelink in background..."
ssh "$BEELINK_HOST" "cd $REMOTE_PATH && (./target/release/niodoo_real_integrated --prompt=\"test rut\" --swarm=\"$SWARM_SIZE\" --output=\"$OUTPUT_FORMAT\" > niodoo_real_integrated.log 2>&1 & PID=\$! && echo \$PID > niodoo_real_integrated.pid && sleep 1 && if ps -p \$PID > /dev/null 2>&1; then echo 'Process started'; else echo 'Process failed to start' >&2; exit 1; fi)"

echo "Waiting for service readiness..."
for i in {1..60}; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Service is ready"
    break
  fi
  sleep 1
  if [ \$i -eq 60 ]; then
    echo "Service failed to become ready" >&2
    exit 1
  fi
done

echo "Running integration tests..."
ssh "$BEELINK_HOST" "cd $REMOTE_PATH && cargo test -p niodoo_real_integrated -- --nocapture"

echo "Stopping binary..."
ssh "$BEELINK_HOST" "cd $REMOTE_PATH && if [ -f niodoo_real_integrated.pid ]; then kill \$(cat niodoo_real_integrated.pid) && rm niodoo_real_integrated.pid; fi"

echo "Deployment complete. Binary: niodoo_real_integrated"
echo "Monitor at: http://beelink:9090 (Prometheus)"
echo "Logs available on Beelink at $REMOTE_PATH"