import json
import os
import sys
import time
from typing import Tuple

import torch
from sentence_transformers import SentenceTransformer

WARMUP_TEXT = "embedding warmup probe"


def resolve_device() -> str:
    requested = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested in {"cuda", "gpu", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("EMBEDDING_DEVICE requested CUDA but no GPU is available")
        return "cuda"

    if requested == "cpu":
        return "cpu"

    raise RuntimeError(f"Unsupported EMBEDDING_DEVICE value: {requested}")


def warmup_model(model: SentenceTransformer) -> float:
    start = time.perf_counter()
    model.encode(WARMUP_TEXT, batch_size=1, show_progress_bar=False)
    duration_ms = (time.perf_counter() - start) * 1000.0
    return duration_ms


def load_model() -> Tuple[SentenceTransformer, str, float]:
    device = resolve_device()
    print(f"[embedding] Loading all-MiniLM-L6-v2 on {device}...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    warmup_ms = warmup_model(model)
    print(
        f"[embedding] Model ready on {device}. Warm-up latency: {warmup_ms:.2f} ms",
        file=sys.stderr,
    )
    return model, device, warmup_ms


def embed_once(model: SentenceTransformer, text: str):
    print("Encoding chunk...", file=sys.stderr)
    embedding = model.encode(text, batch_size=1, show_progress_bar=False)
    print("Chunk done.", file=sys.stderr)
    return embedding.tolist()


def serve():
    model, device, warmup_ms = load_model()
    print(
        f"[embedding] Serving embeddings on {device} (warm-up {warmup_ms:.2f} ms)",
        file=sys.stderr,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request.get("text", "")
            embedding = embed_once(model, text)
            response = {
                "status": "success",
                "embedding": embedding,
                "device": device,
            }
        except Exception as exc:  # pylint: disable=broad-except
            response = {"status": "error", "message": str(exc)}
        print(json.dumps(response), flush=True)


def embed_cli(model: SentenceTransformer, text: str, device: str, warmup_ms: float):
    try:
        embedding = embed_once(model, text)
        print(
            json.dumps(
                {
                    "status": "success",
                    "embedding": embedding,
                    "device": device,
                    "warmup_ms": warmup_ms,
                }
            )
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(json.dumps({"status": "error", "message": str(exc)}))
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Missing command argument"}))
        sys.exit(1)

    command = sys.argv[1]

    if command == "embed":
        if len(sys.argv) < 3:
            print(json.dumps({"status": "error", "message": "Missing text for embedding"}))
            sys.exit(1)
        model, device, warmup_ms = load_model()
        embed_cli(model, sys.argv[2], device, warmup_ms)
    elif command == "serve":
        serve()
    else:
        print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
