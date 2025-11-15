// Build script for protobuf compilation
// This generates Rust code from .proto files using tonic-build

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc_path);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(
            &[
                "../proto/onnx_inference.proto",
                "../proto/topological_data.proto",
            ],
            &["../proto"],
        )?;
    Ok(())
}
