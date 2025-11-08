# NIODOO Python Library

This library exposes NIODOO pipeline components as importable functions for use in agent-generated code executed in sandboxed environments.

## Installation

```bash
cd niodoo_lib/python
pip install -e .
```

## Usage

```python
import niodoo

# Get embedding
embedding = niodoo.embedder.get_embedding("user prompt text")

# Retrieve related memories
memories = niodoo.erag.retrieve(embedding=embedding, top_k=5)

# Analyze topology
signature = niodoo.tcs.analyze(matrix)

# Evaluate compass
outcome = niodoo.compass.evaluate(pad_state, topology)

# Generate text
response = niodoo.generation.generate("prompt")
```

## Note

This library currently contains placeholder implementations. FFI bindings to the Rust backend will be implemented in a future phase.



