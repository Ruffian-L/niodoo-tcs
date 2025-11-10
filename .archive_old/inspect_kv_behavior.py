#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx"

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
outputs = sess.get_outputs()

print("Inputs ({}):".format(len(inputs)))
for idx, meta in enumerate(inputs):
    print(f"  {idx:02d}: {meta.name} -> {meta.shape} {meta.type}")

batch = 1
seq1 = 4
num_layers = 24
num_heads = 2
head_dim = 64

# First inference data
input_feed = {}
input_feed["input_ids"] = np.arange(seq1, dtype=np.int64).reshape(batch, seq1)
input_feed["position_ids"] = np.arange(seq1, dtype=np.int64).reshape(batch, seq1)
input_feed["attention_mask"] = np.ones((batch, seq1), dtype=np.int64)

for layer in range(num_layers):
    key_name = f"past_key_values.{layer}.key"
    value_name = f"past_key_values.{layer}.value"
    zeros = np.zeros((batch, num_heads, 0, head_dim), dtype=np.float32)
    input_feed[key_name] = zeros
    input_feed[value_name] = zeros

print("\nRunning first inference...")
first_outputs = sess.run(None, input_feed)
print("First run produced {} outputs".format(len(first_outputs)))

# Extract kv cache outputs
present = {}
for layer in range(num_layers):
    key_name = f"present.{layer}.key"
    value_name = f"present.{layer}.value"
    present[key_name] = first_outputs[1 + layer * 2]
    present[value_name] = first_outputs[1 + layer * 2 + 1]
    print(f"Layer {layer} key shape {present[key_name].shape}")

# Second inference attempt with a single new token
seq2 = 1
input_ids_2 = np.array([[99]], dtype=np.int64)
position_ids_2 = np.array([[seq1]], dtype=np.int64)
attention_mask_2 = np.ones((batch, seq1 + seq2), dtype=np.int64)

second_feed = {
    "input_ids": input_ids_2,
    "position_ids": position_ids_2,
    "attention_mask": attention_mask_2,
}
for layer in range(num_layers):
    second_feed[f"past_key_values.{layer}.key"] = present[f"present.{layer}.key"]
    second_feed[f"past_key_values.{layer}.value"] = present[f"present.{layer}.value"]

print("\nRunning second inference...")
try:
    second_outputs = sess.run(None, second_feed)
    print("Second run succeeded, logits shape", second_outputs[0].shape)
except Exception as e:
    print("Second run failed:", e)
