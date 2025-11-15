import os

import onnx
from onnx import TensorProto


def tensor_proto_to_str(t: TensorProto) -> str:
    return TensorProto.DataType.Name(t)


def format_shape(dims):
    parts = []
    for d in dims:
        if isinstance(d, int):
            parts.append(str(d))
        else:
            parts.append(str(d) or "?")
    return "[" + ", ".join(parts) + "]"


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(repo_root, "models", "Qwen3-Embedding-4B-ONNX", "model.onnx")
    model_path = os.environ.get("QWEN_MODEL_PATH", default_path)

    print(f"Loading ONNX model from: {model_path}")
    model = onnx.load(model_path)
    g = model.graph

    print("\nModel info:")
    print(f"  producer_name: {model.producer_name}")
    print(f"  producer_version: {model.producer_version}")
    if model.opset_import:
        for opset in model.opset_import:
            print(f"  opset: domain='{opset.domain}' version={opset.version}")

    print("\nInputs:")
    for inp in g.input:
        t = inp.type.tensor_type
        dtype = tensor_proto_to_str(t.elem_type)
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
        print(f"  - name={inp.name}")
        print(f"    dtype={dtype}")
        print(f"    shape={format_shape(dims)}")

    print("\nOutputs:")
    for out in g.output:
        t = out.type.tensor_type
        dtype = tensor_proto_to_str(t.elem_type)
        dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
        print(f"  - name={out.name}")
        print(f"    dtype={dtype}")
        print(f"    shape={format_shape(dims)}")


if __name__ == "__main__":
    main()
