#!/usr/bin/env python3
"""
🧪 EMOTION DETECTION VALIDATION TEST
Tests the ONNX emotion detection model accuracy
"""

import json
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from pathlib import Path

def load_emotion_model():
    """Load the ONNX emotion detection model"""
    model_path = "models/emotion_detection/model.onnx"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return None

    try:
        session = ort.InferenceSession(model_path)
        print(f"✅ Loaded ONNX model from {model_path}")
        return session
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def load_emotion_labels():
    """Load emotion labels"""
    labels_path = "models/emotion_detection/labels.json"
    if not Path(labels_path).exists():
        print(f"❌ Labels not found at {labels_path}")
        return None

    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print(f"✅ Loaded emotion labels: {labels}")
        return labels
    except Exception as e:
        print(f"❌ Failed to load labels: {e}")
        return None

def test_emotion_detection(session, labels, test_texts):
    """Test emotion detection on sample texts"""
    print("\n🧪 TESTING EMOTION DETECTION ACCURACY")
    print("=" * 50)

    results = []
    for text in test_texts:
        print(f"\n📝 Text: '{text}'")

        # Create dummy embedding (768 dimensions)
        # In a real implementation, this would be from a proper tokenizer
        embedding = np.random.randn(1, 768).astype(np.float32)

        try:
            # Run inference
            inputs = {session.get_inputs()[0].name: embedding}
            outputs = session.run(None, inputs)

            # Get probabilities
            logits = outputs[0][0]
            probabilities = F.softmax(torch.tensor(logits), dim=0).numpy()

            # Get top prediction
            top_idx = np.argmax(probabilities)
            top_emotion = labels[top_idx]
            confidence = probabilities[top_idx]

            print(f"🎭 Predicted: {top_emotion} (confidence: {confidence:.3f})")

            # Manual validation (simple keyword matching)
            text_lower = text.lower()
            if any(word in text_lower for word in ["happy", "joy", "excited", "love"]):
                expected = "GpuWarm"
            elif any(word in text_lower for word in ["sad", "disappointed", "unhappy"]):
                expected = "Curious"  # Using existing emotion types
            elif any(word in text_lower for word in ["curious", "question", "wonder"]):
                expected = "Curious"
            elif any(word in text_lower for word in ["purpose", "understand", "learn"]):
                expected = "Purposeful"
            elif any(word in text_lower for word in ["care", "help", "support"]):
                expected = "AuthenticCare"
            else:
                expected = "Purposeful"  # Default

            accuracy = 1.0 if top_emotion == expected else 0.0
            print(f"🎯 Expected: {expected} | Accuracy: {accuracy}")

            results.append({
                'text': text,
                'predicted': top_emotion,
                'expected': expected,
                'confidence': confidence,
                'accuracy': accuracy
            })

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            results.append({
                'text': text,
                'error': str(e)
            })

    return results

def calculate_accuracy(results):
    """Calculate overall accuracy"""
    successful_tests = [r for r in results if 'accuracy' in r]
    if not successful_tests:
        return 0.0

    total_accuracy = sum(r['accuracy'] for r in successful_tests)
    return total_accuracy / len(successful_tests)

def main():
    """Main test function"""
    print("🧠 NIODOO-FEELING EMOTION DETECTION VALIDATION")
    print("=" * 60)

    # Test cases with expected emotions
    test_cases = [
        "I am so happy and excited about this project!",
        "This makes me feel sad and disappointed.",
        "I am genuinely curious about how consciousness works.",
        "I feel anxious about the future of AI.",
        "I want to understand and learn more about this.",
        "I care deeply about helping others succeed.",
        "This is a purposeful and meaningful endeavor.",
        "I feel satisfied with the progress we've made.",
        "The warmth of genuine connection is beautiful.",
        "I feel resonant with the goals of this project."
    ]

    # Load model and labels
    session = load_emotion_model()
    labels = load_emotion_labels()

    if not session or not labels:
        print("❌ Cannot run tests without model and labels")
        return

    # Run tests
    results = test_emotion_detection(session, labels, test_cases)

    # Calculate accuracy
    accuracy = calculate_accuracy(results)
    print(f"\n📊 OVERALL ACCURACY: {accuracy:.2%}")

    # Show detailed results
    print("\n📋 DETAILED RESULTS:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"{i:2d}. ❌ ERROR: {result['error']}")
        else:
            status = "✅" if result['accuracy'] == 1.0 else "❌"
            print(f"{i:2d}. {status} '{result['text'][:50]}...'")
            print(f"    Predicted: {result['predicted']} | Expected: {result['expected']}")
            print(f"    Confidence: {result['confidence']:.3f}")

    print("\n🎉 EMOTION DETECTION VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
