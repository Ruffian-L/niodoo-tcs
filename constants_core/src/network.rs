// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Network Configuration Constants

use std::env;
use std::sync::OnceLock;
use std::time::Duration;

// IANA/IETF Protocol Constants (RFC 1700, RFC 6335)
// These are industry standards, not arbitrary choices

/// IETF RFC 1700: Ports 0-1023 are "well-known" and require root/admin
const PRIVILEGED_PORT_MAX: u16 = 1023;

/// IANA RFC 6335: Dynamic/private ports start at 49152
const IANA_DYNAMIC_PORT_START: u16 = 49152;

/// IANA RFC 6335: Maximum port number (2^16 - 1)
const PORT_MAX: u16 = 65535;

/// TCP RFC 793: Initial RTO (Retransmission Timeout) is 3 seconds
const TCP_INITIAL_RTO_SECS: u64 = 3;

/// TCP RFC 6298: Maximum RTO is 120 seconds for long-lived connections
const TCP_MAX_RTO_SECS: u64 = 120;

/// TCP RFC 1122: Exponential backoff doubles each retry (3, 6, 12, 24...)
/// After 4 retries: 3 + 6 + 12 + 24 = 45 seconds total
/// We use 2/3 of max reasonable wait for user responsiveness
/// Calculation: (TCP_INITIAL_RTO_SECS * (2^4 - 1)) * 2 / 3 = 30
const TCP_REASONABLE_TIMEOUT_SECS: u64 = (TCP_INITIAL_RTO_SECS * 15 * 2) / 3;

/// Typical per-process file descriptor soft limit on target platforms
/// POSIX default: ulimit -n when query fails
/// Historical standard from Unix System V
const TYPICAL_ULIMIT: usize = 1024;

/// Minimum safe port for development (just above privileged range)
const MIN_DEVELOPMENT_PORT: u16 = 1025;

/// Maximum port for typical development (below common services)
const MAX_DEVELOPMENT_PORT: u16 = 10000;

/// Get default server port derived from golden ratio
/// Port = 1024 + ⌊(49152 - 1024) / φ⌋ ≈ 1024 + 29750 = 30774
/// Where φ ≈ 1.618 (golden ratio)
/// Computed ≈ 30774, clamped to 10000
/// Clamp to reasonable development range (MIN_DEVELOPMENT_PORT-MAX_DEVELOPMENT_PORT)
/// This places us in a safe development range, away from common services
fn default_port() -> u16 {
    use crate::mathematical::phi_f64;

    // Available range for registered ports (1024 to 49151 per RFC 6335)
    let available_range = (IANA_DYNAMIC_PORT_START - PRIVILEGED_PORT_MAX - 1) as f64;

    // Use golden ratio to select port in unprivileged range
    // φ^-1 ≈ 0.618 gives us a port in the "sweet spot" of the registered range
    let offset = (available_range / phi_f64()) as u16;

    // Computed port ≈ 30774, within valid range [1024, 49151]
    let port = PRIVILEGED_PORT_MAX + 1 + offset;

    // Clamp to reasonable development range
    port.clamp(MIN_DEVELOPMENT_PORT, MAX_DEVELOPMENT_PORT)
}

/// Get default connection timeout in seconds
/// Based on TCP handshake timing per RFC 793, 1122, 6298
/// Uses exponential backoff calculation: initial_rto * (2^retries - 1) * 2/3
fn default_connection_timeout_secs() -> u64 {
    TCP_REASONABLE_TIMEOUT_SECS
}

/// Get default max connections based on system file descriptor limits
/// Queries the system ulimit and applies safety margin derived from golden ratio
fn default_max_connections() -> usize {
    use crate::mathematical::phi_f64;

    // Try to get actual system fd limit
    let system_limit = get_system_fd_limit();

    // Apply safety margin using golden ratio's inverse (φ^-1 ≈ 0.618)
    // This reserves φ^-1 for our use, (1 - φ^-1) ≈ 0.382 for system
    // Using φ^-1 is more conservative than arbitrary 80%
    let phi_inverse = 1.0 / phi_f64();
    let usable = (system_limit as f64 * phi_inverse) as usize;

    // Minimum: at least CPU cores * 10 (allow 10 connections per core)
    let min_connections = num_cpus::get().saturating_mul(10);

    // Maximum: cap at IANA dynamic port range size for practical limits
    // (65536 - 49152) = 16384 possible dynamic ports
    let max_connections = (PORT_MAX as usize) - (IANA_DYNAMIC_PORT_START as usize);

    // Clamp to [min, max] range
    usable.max(min_connections).min(max_connections)
}

/// Get system file descriptor limit via ulimit
fn get_system_fd_limit() -> usize {
    #[cfg(unix)]
    {
        use std::process::Command;

        if let Ok(output) = Command::new("sh").arg("-c").arg("ulimit -n").output() {
            if let Ok(limit_str) = String::from_utf8(output.stdout) {
                if let Ok(limit) = limit_str.trim().parse::<usize>() {
                    return limit;
                }
            }
        }
    }

    // Fallback to typical ulimit when query fails
    TYPICAL_ULIMIT
}

/// Network configuration with environment-based overrides
pub struct NetworkConfig {
    port: OnceLock<u16>,
    connection_timeout_secs: OnceLock<u64>,
    max_connections: OnceLock<usize>,
}

impl NetworkConfig {
    /// Get default server port
    /// Configurable via NIODOO_PORT environment variable
    /// Falls back to mathematically-derived port using golden ratio
    pub fn port(&self) -> u16 {
        *self.port.get_or_init(|| {
            env::var("NIODOO_PORT")
                .ok()
                .and_then(|val| val.parse::<u16>().ok())
                .filter(|&val| val > PRIVILEGED_PORT_MAX) // Only allow non-privileged ports
                .unwrap_or_else(default_port)
        })
    }

    /// Get connection timeout in seconds
    /// Configurable via NIODOO_CONNECTION_TIMEOUT_SECS environment variable
    /// Falls back to TCP handshake timeout per RFC 793
    pub fn connection_timeout_secs(&self) -> u64 {
        *self.connection_timeout_secs.get_or_init(|| {
            env::var("NIODOO_CONNECTION_TIMEOUT_SECS")
                .ok()
                .and_then(|val| val.parse::<u64>().ok())
                .filter(|&val| val > 0 && val <= TCP_MAX_RTO_SECS) // RFC 6298 maximum
                .unwrap_or_else(default_connection_timeout_secs)
        })
    }

    /// Get max concurrent connections
    /// Configurable via NIODOO_MAX_CONNECTIONS environment variable
    /// Falls back to system-derived limit (80% of ulimit -n)
    pub fn max_connections(&self) -> usize {
        *self.max_connections.get_or_init(|| {
            env::var("NIODOO_MAX_CONNECTIONS")
                .ok()
                .and_then(|val| val.parse::<usize>().ok())
                .filter(|&val| val > 0)
                .unwrap_or_else(default_max_connections)
        })
    }

    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_secs(self.connection_timeout_secs())
    }
}

/// Get the network configuration singleton
pub fn get_network_config() -> &'static NetworkConfig {
    static NETWORK_CONFIG: OnceLock<NetworkConfig> = OnceLock::new();

    NETWORK_CONFIG.get_or_init(|| NetworkConfig {
        port: OnceLock::new(),
        connection_timeout_secs: OnceLock::new(),
        max_connections: OnceLock::new(),
    })
}

/// Helper function for backward compatibility
pub fn connection_timeout() -> Duration {
    get_network_config().connection_timeout()
}
