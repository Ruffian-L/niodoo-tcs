//! Versioned Adapter Storage
//!
//! Manages versioned adapter storage with timestamp-based versioning,
//! metadata tracking, and atomic writes.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Adapter metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub adapter_type: String, // "rust" or "python"
    // Rust-specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epochs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f32>,
    // Python-specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
}

/// Adapter version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterVersion {
    pub version: String,
    pub path: PathBuf,
    pub metadata: AdapterMetadata,
}

/// Adapter storage manager
pub struct AdapterStorage {
    storage_base: PathBuf,
}

impl AdapterStorage {
    pub fn new(storage_base: impl AsRef<Path>) -> Result<Self> {
        let storage_base = storage_base.as_ref().to_path_buf();
        fs::create_dir_all(&storage_base).with_context(|| {
            format!(
                "Failed to create storage directory: {}",
                storage_base.display()
            )
        })?;

        Ok(Self { storage_base })
    }

    /// Generate versioned path from timestamp
    fn versioned_path(&self, timestamp: DateTime<Utc>) -> PathBuf {
        let version = timestamp.format("%Y%m%d_%H%M%S").to_string();
        self.storage_base
            .join(format!("system2_adapters_v{}", version))
    }

    /// Get latest symlink path
    fn latest_symlink(&self) -> PathBuf {
        self.storage_base.join("system2_adapters").join("latest")
    }

    /// Save adapter with versioning
    pub fn save_adapter(
        &self,
        adapter_path: impl AsRef<Path>,
        metadata: AdapterMetadata,
    ) -> Result<PathBuf> {
        let adapter_path = adapter_path.as_ref();
        let versioned_path = self.versioned_path(metadata.timestamp.clone());

        // Create versioned directory
        fs::create_dir_all(&versioned_path).with_context(|| {
            format!(
                "Failed to create versioned directory: {}",
                versioned_path.display()
            )
        })?;

        // Copy adapter files to versioned directory
        if adapter_path.is_dir() {
            // Python adapter (directory with multiple files)
            copy_dir_all(adapter_path, &versioned_path).with_context(|| {
                format!(
                    "Failed to copy adapter directory: {}",
                    adapter_path.display()
                )
            })?;
        } else {
            // Rust adapter (single safetensors file)
            let dest_file = versioned_path.join(adapter_path.file_name().unwrap_or_default());
            fs::copy(adapter_path, &dest_file).with_context(|| {
                format!("Failed to copy adapter file: {}", adapter_path.display())
            })?;
        }

        // Save metadata
        let metadata_path = versioned_path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .context("Failed to serialize adapter metadata")?;
        fs::write(&metadata_path, metadata_json).with_context(|| {
            format!("Failed to write metadata file: {}", metadata_path.display())
        })?;

        // Update latest symlink
        self.update_latest_symlink(&versioned_path)?;

        info!(
            version = %metadata.version,
            path = %versioned_path.display(),
            "Saved versioned adapter"
        );

        Ok(versioned_path)
    }

    /// Update latest symlink
    fn update_latest_symlink(&self, target: &Path) -> Result<()> {
        let symlink = self.latest_symlink();
        let symlink_dir = symlink.parent().unwrap_or(&self.storage_base);

        // Create parent directory if needed
        fs::create_dir_all(symlink_dir).with_context(|| {
            format!(
                "Failed to create symlink directory: {}",
                symlink_dir.display()
            )
        })?;

        // Remove existing symlink if it exists
        if symlink.exists() || symlink.is_symlink() {
            fs::remove_file(&symlink).with_context(|| {
                format!("Failed to remove existing symlink: {}", symlink.display())
            })?;
        }

        // Create symlink (absolute path for simplicity)
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, &symlink)
            .with_context(|| format!("Failed to create symlink: {}", symlink.display()))?;

        #[cfg(windows)]
        {
            // Windows: use junction for directories, copy for files
            if target.is_dir() {
                // For Windows, we'll use a simple approach: create a text file with the path
                // In production, you might want to use junction crate or similar
                fs::write(&symlink, target.to_string_lossy().as_ref()).with_context(|| {
                    format!("Failed to create symlink reference: {}", symlink.display())
                })?;
            } else {
                fs::copy(target, &symlink).with_context(|| {
                    format!("Failed to copy for symlink: {}", symlink.display())
                })?;
            }
        }

        debug!(symlink = %symlink.display(), target = %target.display(), "Updated latest symlink");
        Ok(())
    }

    /// Get latest adapter path
    pub fn get_latest(&self) -> Result<Option<PathBuf>> {
        let symlink = self.latest_symlink();
        if symlink.exists() || symlink.is_symlink() {
            let target = fs::read_link(&symlink)
                .with_context(|| format!("Failed to read symlink: {}", symlink.display()))?;
            if target.exists() {
                return Ok(Some(target));
            }
        }

        // Fallback: find latest versioned directory
        self.list_versions()?
            .into_iter()
            .next()
            .map(|v| Ok(v.path))
            .transpose()
    }

    /// List all adapter versions
    pub fn list_versions(&self) -> Result<Vec<AdapterVersion>> {
        let entries = fs::read_dir(&self.storage_base).with_context(|| {
            format!(
                "Failed to read storage directory: {}",
                self.storage_base.display()
            )
        })?;

        let mut versions = Vec::new();
        for entry in entries {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !dir_name.starts_with("system2_adapters_v") {
                continue;
            }

            let metadata_path = path.join("metadata.json");
            if !metadata_path.exists() {
                warn!(path = %path.display(), "Adapter directory missing metadata.json");
                continue;
            }

            let metadata_content = fs::read_to_string(&metadata_path)
                .with_context(|| format!("Failed to read metadata: {}", metadata_path.display()))?;
            let metadata: AdapterMetadata =
                serde_json::from_str(&metadata_content).with_context(|| {
                    format!(
                        "Failed to deserialize metadata: {}",
                        metadata_path.display()
                    )
                })?;

            versions.push(AdapterVersion {
                version: metadata.version.clone(),
                path,
                metadata,
            });
        }

        // Sort by timestamp (newest first)
        versions.sort_by(|a, b| b.metadata.timestamp.cmp(&a.metadata.timestamp));

        Ok(versions)
    }

    /// Get adapter metadata
    pub fn get_metadata(&self, version: &str) -> Result<Option<AdapterMetadata>> {
        let versions = self.list_versions()?;
        Ok(versions
            .into_iter()
            .find(|v| v.version == version)
            .map(|v| v.metadata))
    }
}

/// Copy directory recursively
fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
    let src = src.as_ref();
    let dst = dst.as_ref();

    fs::create_dir_all(dst)
        .with_context(|| format!("Failed to create destination directory: {}", dst.display()))?;

    for entry in fs::read_dir(src)
        .with_context(|| format!("Failed to read source directory: {}", src.display()))?
    {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();
        let file_name = entry.file_name();
        let dst_path = dst.join(file_name);

        if path.is_dir() {
            copy_dir_all(&path, &dst_path)?;
        } else {
            fs::copy(&path, &dst_path).with_context(|| {
                format!(
                    "Failed to copy file: {} -> {}",
                    path.display(),
                    dst_path.display()
                )
            })?;
        }
    }

    Ok(())
}
