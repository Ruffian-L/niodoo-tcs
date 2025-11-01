use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use blake3;
use chrono::Utc;
use hex;
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use tracing::warn;

static ENV_OVERRIDES: OnceCell<RwLock<HashMap<String, String>>> = OnceCell::new();

fn env_store() -> &'static RwLock<HashMap<String, String>> {
    ENV_OVERRIDES.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn set_env_override<K, V>(key: K, value: V)
where
    K: Into<String>,
    V: Into<String>,
{
    let key = key.into();
    let value = value.into();
    env_store().write().insert(key.clone(), value.clone());

    if let Err(error) = append_config_audit_log(&key, &value) {
        warn!(%key, ?error, "failed to record configuration override");
    }
}

fn append_config_audit_log(key: &str, value: &str) -> Result<()> {
    let path = PathBuf::from("./logs/config_audit.log");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "unable to create config audit directory at {}",
                parent.display()
            )
        })?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("unable to open config audit log at {}", path.display()))?;

    let timestamp = Utc::now().to_rfc3339();
    let digest = blake3::hash(value.as_bytes());
    writeln!(
        file,
        "{timestamp} key={key} value_hash={} char_count={}",
        hex::encode(digest.as_bytes()),
        value.chars().count()
    )?;
    Ok(())
}

pub fn env_value(key: &str) -> Option<String> {
    env_store()
        .read()
        .get(key)
        .cloned()
        .or_else(|| env::var(key).ok())
}

pub fn env_var(key: &str) -> std::result::Result<String, std::env::VarError> {
    if let Some(value) = env_store().read().get(key) {
        return Ok(value.clone());
    }
    env::var(key)
}

pub fn prime_environment() {
    let mut roots: HashSet<PathBuf> = HashSet::new();

    if let Ok(project_root) = env::var("PROJECT_ROOT") {
        if !project_root.trim().is_empty() {
            roots.insert(PathBuf::from(project_root));
        }
    }

    if let Ok(current) = std::env::current_dir() {
        roots.insert(current);
    }

    roots.insert(PathBuf::from("."));

    let env_files = [".env.production", ".env"];
    let mut seen_paths = HashSet::new();

    for root in roots {
        for file in env_files {
            let path = root.join(file);
            if !path.is_file() {
                continue;
            }
            if !seen_paths.insert(path.clone()) {
                continue;
            }
            if let Err(error) = load_env_file(&path) {
                warn!(path = %path.display(), ?error, "failed to load environment file");
            }
        }
    }
}

pub fn env_with_fallback(keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = env_value(key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn load_env_file(path: &Path) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("unable to read env file {}", path.display()))?;

    for (_line_index, line) in contents.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.splitn(2, '=');
        let key = parts.next().unwrap_or("").trim();
        if key.is_empty() {
            continue;
        }
        let raw_value = parts.next().unwrap_or("").trim();
        let value = normalise_env_value(raw_value);
        set_env_override(key, value);
    }

    Ok(())
}

fn normalise_env_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let first = trimmed.as_bytes()[0] as char;
        let last = trimmed.as_bytes()[trimmed.len() - 1] as char;
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return trimmed[1..trimmed.len() - 1].trim().to_string();
        }
    }
    trimmed.trim_end_matches('\r').to_string()
}

