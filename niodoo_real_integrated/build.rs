use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(err) = build_protos() {
        println!("cargo:warning=Skipping proto compilation: {err}");
    }

    // Build federated proto with tonic
    if let Err(err) = build_federated_proto() {
        println!("cargo:warning=Skipping federated proto compilation: {err}");
    }
}

fn build_protos() -> Result<(), Box<dyn Error>> {
    let proto = Path::new("src/proto/niodoo.proto");
    if !proto.exists() {
        return Ok(());
    }

    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    env::set_var("PROTOC", protoc_path);

    let include_dirs = ["src/"];
    match prost_build::Config::new().compile_protos(&[proto], &include_dirs) {
        Ok(()) => Ok(()),
        Err(err) => {
            let out_file = PathBuf::from(std::env::var("OUT_DIR")?).join("niodoo.rs");
            fs::copy("src/proto/niodoo_fallback.rs", &out_file)?;
            Err(Box::new(err))
        }
    }
}

fn build_federated_proto() -> Result<(), Box<dyn Error>> {
    let proto = Path::new("src/federated.proto");
    if !proto.exists() {
        return Ok(());
    }

    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    env::set_var("PROTOC", protoc_path);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&[proto], &["src/"])?;

    Ok(())
}
