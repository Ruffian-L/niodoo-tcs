fn main() {
use tracing::{info, error, warn};
    tracing::info!(
        "Test: {:?}",
        std::path::Path::new("test.md").ends_with(".md")
    );
}
