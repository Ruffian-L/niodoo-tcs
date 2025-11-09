use tcs_parser::{get_ast, graph::ast_to_graph};

fn main() {
    let code = r##"
fn test() {
    let x = 1;
    let y = 2;
}
    "##;
    
    let tree = get_ast(code, "rust").expect("Failed to parse");
    println!("Full AST:");
    println!("{}", tree.root_node().to_sexp());
    
    let graph = ast_to_graph(&tree, code).expect("Failed to build graph");
    
    println!("\nGraph stats:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());
}
