#!/usr/bin/env python3
"""
Tree-sitter parser bridge for tcs-parser

This bypasses the Rust C linker issue by using Python's tree-sitter bindings.
Since Day 2 already requires Python FFI for giotto-tda, we reuse the same
bridge for parsing.

Requirements:
    pip install tree-sitter
"""

import os
from tree_sitter import Language, Parser

# Build language libraries from vendor/ sources
# These are the same sources the Rust crates would use
LANGUAGES_SO = 'build/languages.so'

# Build once if needed
if not os.path.exists(LANGUAGES_SO):
    os.makedirs('build', exist_ok=True)
    Language.build_library(
        LANGUAGES_SO,
        [
            'vendor/tree-sitter-rust',
            'vendor/tree-sitter-python',
        ]
    )

# Load languages
RUST_LANGUAGE = Language(LANGUAGES_SO, 'rust')
PYTHON_LANGUAGE = Language(LANGUAGES_SO, 'python')

def parse_code_to_sexp(code_str: str, language: str) -> str:
    """
    Parses a code string and returns its S-expression (AST) as a string.

    Args:
        code_str: Source code to parse
        language: Language name ("rust" or "python")

    Returns:
        S-expression string representation of the AST

    Raises:
        ValueError: If language is not supported
    """
    parser = Parser()

    if language == "rust":
        parser.set_language(RUST_LANGUAGE)
    elif language == "python":
        parser.set_language(PYTHON_LANGUAGE)
    else:
        raise ValueError(f"Unsupported language: {language}")

    tree = parser.parse(bytes(code_str, "utf8"))
    return tree.root_node.sexp()


if __name__ == "__main__":
    # Standalone test
    test_rust_code = "fn main() { let x = 42; }"
    test_python_code = "def hello():\n    x = 42"

    print("Testing Rust parser:")
    rust_sexp = parse_code_to_sexp(test_rust_code, "rust")
    print(f"  Result: {rust_sexp[:100]}...")

    print("\nTesting Python parser:")
    python_sexp = parse_code_to_sexp(test_python_code, "python")
    print(f"  Result: {python_sexp[:100]}...")

    print("\n✓ Parser bridge working!")
