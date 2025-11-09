// Minimal test to see if tree-sitter links
use tree_sitter::{Parser};
use tree_sitter_rust::LANGUAGE as RUST_LANGUAGE;

fn main() {
    let mut parser = Parser::new();
    parser.set_language(&RUST_LANGUAGE.into()).unwrap();
    println!("Success!");
}
