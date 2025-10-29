fn main() {
    prost_build::compile_protos(&["src/proto/niodoo.proto"], &["src/"]).unwrap();
}
