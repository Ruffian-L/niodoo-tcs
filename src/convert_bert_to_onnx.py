#!/usr/bin/env python3
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from optimum.exporters.onnx import main_export_dynamic
import os

# Load the downloaded model
model_name = "models/bert-emotion"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Export to ONNX with dynamic axes
main_export_dynamic(
    model=model,
    task="text-classification",
    model_name_or_path=model_name,
    output="models/bert-emotion/bert-emotion.onnx",
    opset=14
)

print("✅ BERT model converted to ONNX: models/bert-emotion/bert-emotion.onnx")
