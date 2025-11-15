// Build script for protobuf compilation
// This generates Rust code from .proto files using tonic-build

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/proto")
        .compile(
            &[
                "proto/onnx_inference.proto",
                "proto/topological_data.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}



