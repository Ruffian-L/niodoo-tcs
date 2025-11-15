fn main() {
    // Set environment variable to use system oniguruma library
    println!("cargo:rustc-env=RUSTONIG_SYSTEM_LIBONIG=1");
}
