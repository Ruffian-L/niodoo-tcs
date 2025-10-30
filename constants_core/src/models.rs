// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

use std::path::PathBuf;

pub const QWEN_DEFAULT_PATH: &str = "/models/qwen2.5-7b-instruct-awq"; // Relative to project or env-based

pub fn get_qwen_model_path() -> PathBuf {
    std::env::var("QWEN_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(QWEN_DEFAULT_PATH))
}
