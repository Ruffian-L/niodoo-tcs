#!/bin/bash
set -e

echo "🚀 Starting RL Execution Harness Training"
echo "========================================="

# Kill any existing rl_server processes
echo "🧹 Cleaning up old processes..."
pkill -f "rl_server" || true
sleep 2

# Start Rust execution harness server
echo "📡 Starting Rust execution harness server..."
cd /workspace/Niodoo-Final/niodoo_real_integrated

# Use bash -lc to load environment, then run cargo in background
bash -lc 'cd /workspace/Niodoo-Final/niodoo_real_integrated && cargo run --bin rl_server --features svc --release > /tmp/rl_server.log 2>&1' &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"

# Wait for server to start (check health endpoint)
echo "⏳ Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Server is running!"
        break
    fi
    echo -n "."
    sleep 1
done

# Check if server is actually running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "❌ Server failed to start"
    echo "📄 Server logs:"
    tail -50 /tmp/rl_server.log
    exit 1
fi

echo ""
echo "✅ RL Harness Server is ready!"
echo "📊 Logs: /tmp/rl_server.log"
echo ""

# Start Python RL training
echo "🐍 Starting Python PPO training..."
cd /workspace/Niodoo-Final/niodoo-ai

python3 -c "from niodoo_ai.rl_training import CodeGenerationPPOTrainer; trainer = CodeGenerationPPOTrainer('Qwen/Qwen3-Coder', problem_file='data/rl_training_problems.jsonl'); trainer.train(num_epochs=10, problems_per_epoch=5)"

echo ""
echo "✅ Training complete!"
echo "🎯 Check outputs in: /workspace/Niodoo-Final/niodoo-ai/outputs/rl_training"
