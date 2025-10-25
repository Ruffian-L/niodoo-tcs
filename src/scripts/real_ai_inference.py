import sys
import json
from sentence_transformers import SentenceTransformer


def load_model(device: str = "cpu") -> SentenceTransformer:
    print(f"Loading all-MiniLM-L6-v2 on {device}...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Model ready.", file=sys.stderr)
    return model


def embed_once(model: SentenceTransformer, text: str):
    print("Encoding chunk...", file=sys.stderr)
    embedding = model.encode(text, batch_size=1, show_progress_bar=False)
    print("Chunk done.", file=sys.stderr)
    return embedding.tolist()


def serve():
    model = load_model(device="cpu")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            text = request.get("text", "")
            embedding = embed_once(model, text)
            response = {"status": "success", "embedding": embedding}
        except Exception as exc:  # pylint: disable=broad-except
            response = {"status": "error", "message": str(exc)}
        print(json.dumps(response), flush=True)


def embed_cli(model: SentenceTransformer, text: str):
    try:
        embedding = embed_once(model, text)
        print(json.dumps({"status": "success", "embedding": embedding}))
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
        model = load_model(device="cpu")
        embed_cli(model, sys.argv[2])
    elif command == "serve":
        serve()
    else:
        print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
