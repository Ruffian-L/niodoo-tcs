// build.rs - Force explicit linking of tree-sitter
//
// The tree-sitter crate compiles libtree-sitter.a but sometimes
// the linker doesn't pull it in properly.

fn main() {
    // Force explicit static linking of tree-sitter
    println!("cargo:rustc-link-lib=static=tree-sitter");
}
