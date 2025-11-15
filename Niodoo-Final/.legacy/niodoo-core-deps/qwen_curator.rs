#![allow(dead_code)]
#![allow(unused_imports)]

mod legacy_impl {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../archive/legacy/src/qwen_curator.rs.legacy"
    ));
}

#[deprecated(note = "Legacy stub module; migrate to niodoo_real_integrated::curator")]
pub use legacy_impl::*;

//! Re-export the production QLoRA curator from niodoo-core.

pub use niodoo_core::qwen_curator::*;

