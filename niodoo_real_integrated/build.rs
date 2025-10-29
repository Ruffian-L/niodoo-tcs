use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/proto/niodoo.proto");
    println!("cargo:rerun-if-changed=src/federated.proto");
    if let Err(err) = build_protos() {
        println!("cargo:warning=Skipping proto compilation: {err}");
    }

    // Build federated proto with tonic
    if let Err(err) = build_federated_proto() {
        println!("cargo:warning=Skipping federated proto compilation: {err}");
    }
}

fn build_protos() -> Result<(), Box<dyn Error>> {
    let path = Path::new("src/proto/niodoo.proto");
    if !path.exists() {
        return Ok(());
    }

    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;

    let include_dirs = ["src/"];
    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc_path);

    match config.compile_protos(&[path], &include_dirs) {
        Ok(()) => Ok(()),
        Err(err) => {
            let out_file = PathBuf::from(env::var("OUT_DIR")?).join("niodoo.rs");
            fs::copy("src/proto/niodoo_fallback.rs", &out_file)?;
            Err(Box::new(err))
        }
    }
}

fn build_federated_proto() -> Result<(), Box<dyn Error>> {
    let path = Path::new("src/federated.proto");
    if !path.exists() {
        return Ok(());
    }

    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;

    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc_path);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos_with_config(config, &[path], &["src/"])?;

    Ok(())
}
