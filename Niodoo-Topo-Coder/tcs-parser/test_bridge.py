#!/usr/bin/env python3
"""
Simple test to verify Rust→Python FFI bridge works
"""

def hello_from_python(name: str) -> str:
    return f"Hello {name} from Python! FFI bridge working."

if __name__ == "__main__":
    print(hello_from_python("Rust"))
