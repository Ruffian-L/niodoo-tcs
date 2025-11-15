use std::collections::VecDeque;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use hex;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::to_string as to_json_string;
use tracing::{debug, error, warn};

use crate::config::{RuntimeConfig, SecurityConfig};

pub struct PromptSecurityManager {
    config: SecurityConfig,
    rate_limiter: RateLimiter,
    filter: ContentFilter,
    audit: AuditLogger,
}

impl PromptSecurityManager {
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let rate_limiter = RateLimiter::new(
            config.rate_limit_window_secs,
            config.rate_limit_max_requests,
        );
        let filter = ContentFilter::new(&config.banned_patterns)?;
        let audit = AuditLogger::new(&config.audit_log_path)?;
        audit.log_bootstrap(&config);

        Ok(Self {
            config,
            rate_limiter,
            filter,
            audit,
        })
    }

    pub fn enforce_prompt(&self, raw_prompt: &str) -> Result<String> {
        if raw_prompt.trim().is_empty() {
            self.audit.log_prompt_rejected("empty_raw", raw_prompt);
            return Err(anyhow!("prompt is empty"));
        }

        if let Err(err) = self.rate_limiter.check() {
            self.audit.log_rate_limit_violation(
                self.config.rate_limit_max_requests,
                self.config.rate_limit_window_secs,
            );
            return Err(err);
        }

        let sanitized = Sanitizer::sanitize(raw_prompt, self.config.allow_control_chars);
        if sanitized.trim().is_empty() {
            self.audit
                .log_prompt_rejected("empty_after_sanitize", raw_prompt);
            return Err(anyhow!("prompt rejected: empty after sanitisation"));
        }

        let char_count = sanitized.chars().count();
        if char_count > self.config.prompt_max_chars {
            self.audit.log_prompt_rejected("length", &sanitized);
            return Err(anyhow!(
                "prompt exceeds maximum length of {} characters (received {})",
                self.config.prompt_max_chars,
                char_count
            ));
        }

        if let Err(err) = self.filter.validate(&sanitized) {
            self.audit.log_prompt_rejected("content_filter", &sanitized);
            return Err(err);
        }

        debug!(chars = char_count, "prompt accepted after security checks");
        self.audit.log_prompt_accept(&sanitized, char_count);
        Ok(sanitized)
    }

    pub fn audit_config_snapshot(&self, config: &RuntimeConfig) {
        if let Err(error) = self.audit.log_config_snapshot(config) {
            warn!(
                ?error,
                "failed to record config snapshot to security audit log"
            );
        }
    }
}

struct RateLimiter {
    window: Duration,
    max_requests: u32,
    enabled: bool,
    timestamps: Mutex<VecDeque<Instant>>,
}

impl RateLimiter {
    fn new(window_secs: u64, max_requests: u32) -> Self {
        Self {
            window: Duration::from_secs(window_secs.max(1)),
            max_requests,
            enabled: max_requests > 0,
            timestamps: Mutex::new(VecDeque::new()),
        }
    }

    fn check(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut timestamps = self.timestamps.lock();
        let now = Instant::now();
        while let Some(front) = timestamps.front() {
            if now.duration_since(*front) > self.window {
                timestamps.pop_front();
            } else {
                break;
            }
        }

        if timestamps.len() as u32 >= self.max_requests {
            return Err(anyhow!(
                "prompt rate limit exceeded: max {} requests each {:?}",
                self.max_requests,
                self.window
            ));
        }

        timestamps.push_back(now);
        Ok(())
    }
}

struct ContentFilter {
    patterns: Vec<Regex>,
}

impl ContentFilter {
    fn new(patterns: &[String]) -> Result<Self> {
        let mut compiled = Vec::new();
        for pattern in patterns {
            if pattern.trim().is_empty() {
                continue;
            }
            let regex = Regex::new(pattern)
                .with_context(|| format!("invalid security filter pattern '{pattern}'"))?;
            compiled.push(regex);
        }
        Ok(Self { patterns: compiled })
    }

    fn validate(&self, prompt: &str) -> Result<()> {
        for pattern in &self.patterns {
            if pattern.is_match(prompt) {
                return Err(anyhow!(
                    "prompt rejected by security filter (pattern: {})",
                    pattern.as_str()
                ));
            }
        }
        Ok(())
    }
}

struct Sanitizer;

impl Sanitizer {
    fn sanitize(input: &str, allow_control_chars: bool) -> String {
        let mut sanitized = String::with_capacity(input.len());
        for ch in input.chars() {
            match ch {
                '\r' => sanitized.push('\n'),
                '\n' | '\t' => sanitized.push(ch),
                _ if !allow_control_chars && ch.is_control() => continue,
                other => sanitized.push(other),
            }
        }
        sanitized
    }
}

struct AuditLogger {
    file: Mutex<File>,
    path: PathBuf,
}

impl AuditLogger {
    fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            create_dir_all(parent).with_context(|| {
                format!(
                    "unable to create security audit directory at {}",
                    parent.display()
                )
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("unable to open security audit log at {}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
            path,
        })
    }

    fn log_bootstrap(&self, config: &SecurityConfig) {
        let line = format!(
            "{} event=security_bootstrap prompt_max_chars={} rate_limit_window_secs={} rate_limit_max_requests={} allow_control_chars={}",
            Utc::now().to_rfc3339(),
            config.prompt_max_chars,
            config.rate_limit_window_secs,
            config.rate_limit_max_requests,
            config.allow_control_chars
        );
        if let Err(error) = self.append_line(&line) {
            error!(?error, path = %self.path.display(), "failed to record security bootstrap");
        }
    }

    fn log_prompt_accept(&self, prompt: &str, char_count: usize) {
        let digest = blake3::hash(prompt.as_bytes());
        let line = format!(
            "{} event=prompt status=accepted hash={} char_count={}",
            Utc::now().to_rfc3339(),
            hex::encode(digest.as_bytes()),
            char_count
        );
        if let Err(error) = self.append_line(&line) {
            error!(?error, path = %self.path.display(), "failed to record accepted prompt");
        }
    }

    fn log_prompt_rejected(&self, reason: &str, prompt: &str) {
        let digest = blake3::hash(prompt.as_bytes());
        let line = format!(
            "{} event=prompt status=rejected reason={} hash={} char_count={}",
            Utc::now().to_rfc3339(),
            reason,
            hex::encode(digest.as_bytes()),
            prompt.chars().count()
        );
        if let Err(error) = self.append_line(&line) {
            error!(?error, path = %self.path.display(), reason, "failed to record rejected prompt");
        }
    }

    fn log_rate_limit_violation(&self, max_requests: u32, window_secs: u64) {
        let line = format!(
            "{} event=prompt status=rate_limited max_requests={} window_secs={}",
            Utc::now().to_rfc3339(),
            max_requests,
            window_secs
        );
        if let Err(error) = self.append_line(&line) {
            error!(?error, path = %self.path.display(), "failed to record rate limit violation");
        }
    }

    fn log_config_snapshot(&self, config: &RuntimeConfig) -> Result<()> {
        let json =
            to_json_string(config).context("failed to serialise runtime config for audit")?;
        let digest = blake3::hash(json.as_bytes());
        let line = format!(
            "{} event=config_snapshot hash={} prompt_max_chars={} rate_limit_window_secs={} rate_limit_max_requests={}",
            Utc::now().to_rfc3339(),
            hex::encode(digest.as_bytes()),
            config.security.prompt_max_chars,
            config.security.rate_limit_window_secs,
            config.security.rate_limit_max_requests
        );
        self.append_line(&line)
    }

    fn append_line(&self, line: &str) -> Result<()> {
        let mut file = self.file.lock();
        writeln!(file, "{}", line).with_context(|| {
            format!(
                "failed to append to security audit log at {}",
                self.path.display()
            )
        })?;
        file.flush().with_context(|| {
            format!(
                "failed to flush security audit log at {}",
                self.path.display()
            )
        })?;
        Ok(())
    }
}
