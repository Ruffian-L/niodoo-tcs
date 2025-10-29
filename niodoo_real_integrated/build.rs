use std::env;
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    const PROTO_FILE: &str = "src/federated.proto";

    println!("cargo:rerun-if-changed={PROTO_FILE}");
    if Path::new(PROTO_FILE).exists() {
        let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
        env::set_var("PROTOC", protoc_path);

        if let Err(err) = tonic_build::configure()
            .build_server(false)
            .compile(&[PROTO_FILE], &["src"])
        {
            println!("cargo:warning=skipping proto compile: {err}");
        }
    }

    Ok(())
}
