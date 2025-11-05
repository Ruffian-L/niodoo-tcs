//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Hardware monitoring collector for Silicon Synapse
//!
//! This module implements hardware monitoring for GPU, CPU, and memory metrics
//! using platform-specific APIs (NVML for NVIDIA, ROCm for AMD, sysinfo for CPU).

use nvml_wrapper::Nvml;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::silicon_synapse::config::HardwareConfig;
use crate::silicon_synapse::plugins::collector::{
    CollectedMetrics, Collector, CollectorConfig, CollectorError, CollectorHealth, CollectorResult,
    CollectorStats, MetricValue,
};
use crate::silicon_synapse::telemetry_bus::{TelemetryEvent, TelemetrySender};
use crate::silicon_synapse::SiliconSynapseError;
use async_trait::async_trait;

/// Hardware monitoring collector
pub struct HardwareCollector {
    config: HardwareConfig,
    collector_config: CollectorConfig,
    telemetry_sender: TelemetrySender,
    is_running: Arc<RwLock<bool>>,
    monitor_task: Option<tokio::task::JoinHandle<()>>,
    nvidia_monitor: Option<NvidiaMonitor>,
    amd_monitor: Option<AmdMonitor>,
}

/// Hardware metrics data structure
#[derive(Debug, Clone)]
pub struct HardwareMetrics {
    pub timestamp: Instant,
    pub gpu_temperature: Option<f32>,
    pub gpu_power: Option<f32>,
    pub gpu_utilization: Option<f32>,
    pub gpu_memory_used: Option<u64>,
    pub gpu_memory_total: Option<u64>,
    pub gpu_fan_speed: Option<f32>,
    pub cpu_temperature: Option<f32>,
    pub cpu_utilization: Option<f32>,
    pub system_memory_used: Option<u64>,
    pub system_memory_total: Option<u64>,
}

/// Hardware monitoring error types
#[derive(Debug, thiserror::Error)]
pub enum HardwareMonitorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("NVML error: {0}")]
    Nvml(nvml_wrapper::error::NvmlError),
    #[error("GPU monitoring error: {0}")]
    GpuMonitoring(String),
    #[error("CPU monitoring error: {0}")]
    CpuMonitoring(String),
    #[error("Memory monitoring error: {0}")]
    MemoryMonitoring(String),
    #[error("System info error: {0}")]
    SystemInfo(String),
}

impl From<HardwareMonitorError> for CollectorError {
    fn from(err: HardwareMonitorError) -> Self {
        CollectorError::HardwareAccessError(err.to_string())
    }
}

/// Hardware monitor trait for platform-specific implementations
pub trait HardwareMonitor: Send + Sync {
    fn collect_metrics(&self) -> Result<HardwareMetrics, HardwareMonitorError>;
    fn is_available(&self) -> bool;
    fn name(&self) -> &str;
}

/// NVIDIA GPU monitor using NVML
#[derive(Clone)]
pub struct NvidiaMonitor {
    nvml: Option<Arc<Nvml>>,
    device_count: u32,
}

/// AMD GPU monitor using sysfs (ROCm SMI compatible)
#[derive(Clone)]
pub struct AmdMonitor {
    adapters: Arc<Vec<AmdGpuAdapter>>,
}

#[derive(Clone)]
pub struct IntelMonitor {
    adapters: Arc<Vec<IntelGpuAdapter>>,
}

#[derive(Debug, Clone)]
struct AmdGpuAdapter {
    card_path: PathBuf,
    hwmon_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct IntelGpuAdapter {
    card_path: PathBuf,
    hwmon_path: Option<PathBuf>,
}

impl AmdMonitor {
    pub fn new() -> Result<Self, HardwareMonitorError> {
        Self::with_root(Path::new("/sys/class/drm"))
    }

    fn with_root(root: &Path) -> Result<Self, HardwareMonitorError> {
        let mut adapters = Vec::new();

        for entry in fs::read_dir(root)? {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };

            let file_name = entry.file_name();
            if !file_name.to_string_lossy().starts_with("card") {
                continue;
            }

            let card_path = entry.path();
            let vendor = fs::read_to_string(card_path.join("device/vendor")).unwrap_or_default();
            if vendor.trim() != "0x1002" {
                continue;
            }

            let hwmon_path = card_path
                .join("device/hwmon")
                .read_dir()
                .ok()
                .and_then(|mut hwmons| hwmons.find_map(|hwmon| hwmon.ok().map(|h| h.path())));

            adapters.push(AmdGpuAdapter {
                card_path,
                hwmon_path,
            });
        }

        Ok(Self {
            adapters: Arc::new(adapters),
        })
    }

    fn collect_from_adapter(
        adapter: &AmdGpuAdapter,
    ) -> Result<
        (
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<u64>,
            Option<u64>,
            Option<f32>,
        ),
        HardwareMonitorError,
    > {
        collect_common_gpu_metrics(
            &adapter.card_path,
            adapter.hwmon_path.as_ref().map(Path::new),
        )
    }
}

impl IntelMonitor {
    pub fn new() -> Result<Self, HardwareMonitorError> {
        Self::with_root(Path::new("/sys/class/drm"))
    }

    fn with_root(root: &Path) -> Result<Self, HardwareMonitorError> {
        let mut adapters = Vec::new();

        for entry in fs::read_dir(root)? {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };

            let file_name = entry.file_name();
            if !file_name.to_string_lossy().starts_with("card") {
                continue;
            }

            let card_path = entry.path();
            let vendor = fs::read_to_string(card_path.join("device/vendor")).unwrap_or_default();
            if vendor.trim() != "0x8086" {
                continue;
            }

            let hwmon_path = card_path
                .join("device/hwmon")
                .read_dir()
                .ok()
                .and_then(|mut hwmons| hwmons.find_map(|hwmon| hwmon.ok().map(|h| h.path())));

            adapters.push(IntelGpuAdapter {
                card_path,
                hwmon_path,
            });
        }

        Ok(Self {
            adapters: Arc::new(adapters),
        })
    }

    fn collect_from_adapter(
        adapter: &IntelGpuAdapter,
    ) -> Result<
        (
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<u64>,
            Option<u64>,
            Option<f32>,
        ),
        HardwareMonitorError,
    > {
        collect_common_gpu_metrics(
            &adapter.card_path,
            adapter.hwmon_path.as_ref().map(Path::new),
        )
    }
}

fn collect_common_gpu_metrics(
    card_path: &Path,
    hwmon_path: Option<&Path>,
) -> Result<
    (
        Option<f32>,
        Option<f32>,
        Option<f32>,
        Option<u64>,
        Option<u64>,
        Option<f32>,
    ),
    HardwareMonitorError,
> {
    let utilization = read_u32(card_path.join("device/gpu_busy_percent"))
        .ok()
        .map(|value| value as f32);
    let vram_used = read_u64(card_path.join("device/mem_info_vram_used")).ok();
    let vram_total = read_u64(card_path.join("device/mem_info_vram_total")).ok();

    let (temperature, power, fan_speed_percent) = match hwmon_path {
        Some(hwmon) => {
            let temperature = read_u32(hwmon.join("temp1_input"))
                .ok()
                .map(|value| value as f32 / 1000.0);
            let power = read_u64(hwmon.join("power1_average"))
                .ok()
                .map(|value| value as f32 / 1_000_000.0);
            let fan_speed_percent = match (
                read_u32(hwmon.join("fan1_input")).ok(),
                read_u32(hwmon.join("fan1_max")).ok(),
            ) {
                (Some(current), Some(maximum)) if maximum > 0 => {
                    Some((current as f32 / maximum as f32) * 100.0)
                }
                (Some(current), None) if current > 0 => Some((current as f32).min(100.0)),
                _ => None,
            };

            (temperature, power, fan_speed_percent)
        }
        None => (None, None, None),
    };

    Ok((
        temperature,
        power,
        utilization,
        vram_used,
        vram_total,
        fan_speed_percent,
    ))
}

impl HardwareMonitor for IntelMonitor {
    fn collect_metrics(&self) -> Result<HardwareMetrics, HardwareMonitorError> {
        let timestamp = Instant::now();

        if let Some(adapter) = self.adapters.first() {
            let (
                gpu_temperature,
                gpu_power,
                gpu_utilization,
                gpu_memory_used,
                gpu_memory_total,
                gpu_fan_speed,
            ) = Self::collect_from_adapter(adapter)?;

            Ok(HardwareMetrics {
                timestamp,
                gpu_temperature,
                gpu_power,
                gpu_utilization,
                gpu_memory_used,
                gpu_memory_total,
                gpu_fan_speed,
                cpu_temperature: None,
                cpu_utilization: None,
                system_memory_used: None,
                system_memory_total: None,
            })
        } else {
            Ok(HardwareMetrics {
                timestamp,
                gpu_temperature: None,
                gpu_power: None,
                gpu_utilization: None,
                gpu_memory_used: None,
                gpu_memory_total: None,
                gpu_fan_speed: None,
                cpu_temperature: None,
                cpu_utilization: None,
                system_memory_used: None,
                system_memory_total: None,
            })
        }
    }

    fn is_available(&self) -> bool {
        !self.adapters.is_empty()
    }

    fn name(&self) -> &str {
        "Intel i915"
    }
}

impl HardwareMonitor for AmdMonitor {
    fn collect_metrics(&self) -> Result<HardwareMetrics, HardwareMonitorError> {
        let timestamp = Instant::now();

        if let Some(adapter) = self.adapters.first() {
            let (
                gpu_temperature,
                gpu_power,
                gpu_utilization,
                gpu_memory_used,
                gpu_memory_total,
                gpu_fan_speed,
            ) = Self::collect_from_adapter(adapter)?;

            Ok(HardwareMetrics {
                timestamp,
                gpu_temperature,
                gpu_power,
                gpu_utilization,
                gpu_memory_used,
                gpu_memory_total,
                gpu_fan_speed,
                cpu_temperature: None,
                cpu_utilization: None,
                system_memory_used: None,
                system_memory_total: None,
            })
        } else {
            Ok(HardwareMetrics {
                timestamp,
                gpu_temperature: None,
                gpu_power: None,
                gpu_utilization: None,
                gpu_memory_used: None,
                gpu_memory_total: None,
                gpu_fan_speed: None,
                cpu_temperature: None,
                cpu_utilization: None,
                system_memory_used: None,
                system_memory_total: None,
            })
        }
    }

    fn is_available(&self) -> bool {
        !self.adapters.is_empty()
    }

    fn name(&self) -> &str {
        "AMD ROCm"
    }
}

fn read_u32(path: PathBuf) -> Result<u32, std::num::ParseIntError> {
    let content = fs::read_to_string(path).unwrap_or_default();
    let trimmed = content.trim();
    trimmed.parse::<u32>()
}

fn read_u64(path: PathBuf) -> Result<u64, std::num::ParseIntError> {
    let content = fs::read_to_string(path).unwrap_or_default();
    let trimmed = content.trim();
    trimmed.parse::<u64>()
}

impl NvidiaMonitor {
    /// Create a new NVIDIA monitor
    pub fn new() -> Result<Self, HardwareMonitorError> {
        match Nvml::init() {
            Ok(nvml) => {
                let device_count = match nvml.device_count() {
                    Ok(count) => count,
                    Err(e) => {
                        warn!("Failed to get NVIDIA device count: {}", e);
                        return Ok(Self {
                            nvml: Some(Arc::new(nvml)),
                            device_count: 0,
                        });
                    }
                };

                info!("NVIDIA monitor initialized with {} devices", device_count);
                Ok(Self {
                    nvml: Some(Arc::new(nvml)),
                    device_count,
                })
            }
            Err(e) => {
                warn!("NVIDIA NVML not available: {}", e);
                Ok(Self {
                    nvml: None,
                    device_count: 0,
                })
            }
        }
    }

    /// Collect GPU metrics from NVIDIA devices
    fn collect_nvidia_metrics(
        &self,
    ) -> Result<
        (
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<u64>,
            Option<u64>,
            Option<f32>,
        ),
        HardwareMonitorError,
    > {
        let nvml = match &self.nvml {
            Some(nvml) => nvml.as_ref(),
            None => return Ok((None, None, None, None, None, None)),
        };

        if self.device_count == 0 {
            return Ok((None, None, None, None, None, None));
        }

        // Use the first device for metrics (can be extended for multi-GPU)
        let device = match nvml.device_by_index(0) {
            Ok(device) => device,
            Err(e) => {
                debug!("Failed to get NVIDIA device 0: {}", e);
                return Ok((None, None, None, None, None, None));
            }
        };

        // Collect temperature
        let temperature =
            match device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu) {
                Ok(temp) => Some(temp as f32),
                Err(e) => {
                    debug!("Failed to get GPU temperature: {}", e);
                    None
                }
            };

        // Collect power usage
        let power = match device.power_usage() {
            Ok(power_mw) => Some(power_mw as f32 / 1000.0), // Convert mW to W
            Err(e) => {
                debug!("Failed to get GPU power usage: {}", e);
                None
            }
        };

        // Collect utilization rates
        let utilization = match device.utilization_rates() {
            Ok(rates) => Some(rates.gpu as f32),
            Err(e) => {
                debug!("Failed to get GPU utilization: {}", e);
                None
            }
        };

        // Collect memory info
        let (memory_used, memory_total) = match device.memory_info() {
            Ok(mem_info) => (Some(mem_info.used), Some(mem_info.total)),
            Err(e) => {
                debug!("Failed to get GPU memory info: {}", e);
                (None, None)
            }
        };

        // Collect fan speed
        let fan_speed = match device.fan_speed(0) {
            Ok(speed) => Some(speed as f32),
            Err(e) => {
                debug!("Failed to get GPU fan speed: {}", e);
                None
            }
        };

        Ok((
            temperature,
            power,
            utilization,
            memory_used,
            memory_total,
            fan_speed,
        ))
    }
}

impl HardwareMonitor for NvidiaMonitor {
    fn collect_metrics(&self) -> Result<HardwareMetrics, HardwareMonitorError> {
        let timestamp = Instant::now();
        let (
            gpu_temperature,
            gpu_power,
            gpu_utilization,
            gpu_memory_used,
            gpu_memory_total,
            gpu_fan_speed,
        ) = self.collect_nvidia_metrics()?;

        Ok(HardwareMetrics {
            timestamp,
            gpu_temperature,
            gpu_power,
            gpu_utilization,
            gpu_memory_used,
            gpu_memory_total,
            gpu_fan_speed,
            cpu_temperature: None, // CPU metrics handled separately
            cpu_utilization: None,
            system_memory_used: None,
            system_memory_total: None,
        })
    }

    fn is_available(&self) -> bool {
        self.nvml.is_some() && self.device_count > 0
    }

    fn name(&self) -> &str {
        "NVIDIA NVML"
    }
}

impl HardwareCollector {
    /// Create a new hardware collector
    pub fn new(
        config: HardwareConfig,
        telemetry_sender: TelemetrySender,
    ) -> Result<Self, SiliconSynapseError> {
        // Initialize NVIDIA monitor if enabled
        let nvidia_monitor = if config.gpu.enable_nvidia {
            match NvidiaMonitor::new() {
                Ok(monitor) => {
                    if monitor.is_available() {
                        info!("NVIDIA monitor initialized successfully");
                        Some(monitor)
                    } else {
                        warn!("NVIDIA monitor not available");
                        None
                    }
                }
                Err(e) => {
                    warn!("Failed to initialize NVIDIA monitor: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let amd_monitor = if config.gpu.enable_amd {
            match AmdMonitor::new() {
                Ok(monitor) => {
                    if monitor.is_available() {
                        info!("AMD monitor initialized successfully");
                        Some(monitor)
                    } else {
                        debug!("AMD monitor not available");
                        None
                    }
                }
                Err(e) => {
                    warn!("Failed to initialize AMD monitor: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Create CollectorConfig from HardwareConfig
        let collector_config = CollectorConfig {
            id: "hardware".to_string(),
            name: "Hardware Monitor".to_string(),
            interval: Duration::from_millis(config.collection_interval_ms),
            enabled: config.enabled,
            parameters: std::collections::HashMap::new(),
        };

        Ok(Self {
            config,
            collector_config,
            telemetry_sender,
            is_running: Arc::new(RwLock::new(false)),
            monitor_task: None,
            nvidia_monitor,
            amd_monitor,
        })
    }

    /// Internal start implementation (used by both public API and trait)
    async fn start_internal(&mut self) -> Result<(), SiliconSynapseError> {
        if *self.is_running.read().await {
            return Err(SiliconSynapseError::HardwareMonitor(
                "Hardware collector is already running".to_string(),
            ));
        }

        info!("Starting hardware collector");

        let config = self.config.clone();
        let telemetry_sender = self.telemetry_sender.clone();
        let is_running = self.is_running.clone();
        let nvidia_monitor = self.nvidia_monitor.clone();
        let amd_monitor = self.amd_monitor.clone();

        let task = tokio::spawn(async move {
            Self::monitor_loop(
                config,
                telemetry_sender,
                is_running,
                nvidia_monitor,
                amd_monitor,
            )
            .await;
        });

        self.monitor_task = Some(task);
        *self.is_running.write().await = true;

        Ok(())
    }

    /// Internal stop implementation (used by both public API and trait)
    async fn stop_internal(&mut self) -> Result<(), SiliconSynapseError> {
        if !*self.is_running.read().await {
            return Ok(());
        }

        info!("Stopping hardware collector");

        *self.is_running.write().await = false;

        if let Some(task) = self.monitor_task.take() {
            task.abort();
        }

        Ok(())
    }

    /// Start the hardware monitoring loop (public API)
    pub async fn start(&mut self) -> Result<(), SiliconSynapseError> {
        self.start_internal().await
    }

    /// Stop the hardware monitoring loop (public API)
    pub async fn stop(&mut self) -> Result<(), SiliconSynapseError> {
        self.stop_internal().await
    }

    /// Main monitoring loop
    async fn monitor_loop(
        config: HardwareConfig,
        telemetry_sender: TelemetrySender,
        is_running: Arc<RwLock<bool>>,
        nvidia_monitor: Option<NvidiaMonitor>,
        amd_monitor: Option<AmdMonitor>,
    ) {
        let mut interval = interval(Duration::from_millis(config.collection_interval_ms));

        while *is_running.read().await {
            interval.tick().await;

            match Self::collect_hardware_metrics(&config, &nvidia_monitor, &amd_monitor).await {
                Ok(metrics) => {
                    let event = TelemetryEvent::HardwareMetrics {
                        timestamp: std::time::SystemTime::now(),
                        gpu_temp_celsius: metrics.gpu_temperature,
                        gpu_power_watts: metrics.gpu_power,
                        gpu_fan_speed_percent: metrics.gpu_fan_speed,
                        vram_used_bytes: metrics.gpu_memory_used,
                        vram_total_bytes: metrics.gpu_memory_total,
                        gpu_utilization_percent: metrics.gpu_utilization,
                        cpu_utilization_percent: metrics.cpu_utilization.unwrap_or(0.0),
                        ram_used_bytes: metrics.system_memory_used.unwrap_or(0),
                    };

                    if let Err(e) = telemetry_sender.try_send(event) {
                        warn!("Failed to send hardware metrics: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to collect hardware metrics: {}", e);
                }
            }
        }
    }

    /// Collect hardware metrics using available system APIs
    async fn collect_hardware_metrics(
        config: &HardwareConfig,
        nvidia_monitor: &Option<NvidiaMonitor>,
        amd_monitor: &Option<AmdMonitor>,
    ) -> Result<HardwareMetrics, HardwareMonitorError> {
        let timestamp = Instant::now();

        // Collect GPU metrics using available monitors
        let (
            gpu_temperature,
            gpu_power,
            gpu_utilization,
            gpu_memory_used,
            gpu_memory_total,
            gpu_fan_speed,
        ) = Self::collect_gpu_metrics(config, nvidia_monitor, amd_monitor).await?;

        // Collect CPU metrics
        let (cpu_temperature, cpu_utilization) = if config.cpu.enabled {
            Self::collect_cpu_metrics(config).await?
        } else {
            (None, None)
        };

        // Collect memory metrics
        let (system_memory_used, system_memory_total) = if config.memory.enabled {
            Self::collect_memory_metrics(config).await?
        } else {
            (None, None)
        };

        Ok(HardwareMetrics {
            timestamp,
            gpu_temperature,
            gpu_power,
            gpu_utilization,
            gpu_memory_used,
            gpu_memory_total,
            gpu_fan_speed,
            cpu_temperature,
            cpu_utilization,
            system_memory_used,
            system_memory_total,
        })
    }

    /// Collect GPU metrics using available APIs
    async fn collect_gpu_metrics(
        config: &HardwareConfig,
        nvidia_monitor: &Option<NvidiaMonitor>,
        amd_monitor: &Option<AmdMonitor>,
    ) -> Result<
        (
            Option<f32>,
            Option<f32>,
            Option<f32>,
            Option<u64>,
            Option<u64>,
            Option<f32>,
        ),
        HardwareMonitorError,
    > {
        let mut gpu_temperature = None;
        let mut gpu_power = None;
        let mut gpu_utilization = None;
        let mut gpu_memory_used = None;
        let mut gpu_memory_total = None;
        let mut gpu_fan_speed = None;

        if !(config.gpu.enable_nvidia || config.gpu.enable_amd || config.gpu.enable_intel) {
            return Ok((None, None, None, None, None, None));
        }

        // Collect NVIDIA GPU metrics if available
        if config.gpu.enable_nvidia {
            if let Some(monitor) = nvidia_monitor {
                match monitor.collect_nvidia_metrics() {
                    Ok((temp, power, util, mem_used, mem_total, fan)) => {
                        gpu_temperature = temp;
                        gpu_power = power;
                        gpu_utilization = util;
                        gpu_memory_used = mem_used;
                        gpu_memory_total = mem_total;
                        gpu_fan_speed = fan;
                        debug!("Collected NVIDIA GPU metrics successfully");
                    }
                    Err(e) => {
                        debug!("Failed to collect NVIDIA GPU metrics: {}", e);
                    }
                }
            } else {
                debug!("NVIDIA monitor not available");
            }
        }

        if config.gpu.enable_amd {
            if let Some(monitor) = amd_monitor {
                match monitor.collect_metrics() {
                    Ok(metrics) => {
                        gpu_temperature = gpu_temperature.or(metrics.gpu_temperature);
                        gpu_power = gpu_power.or(metrics.gpu_power);
                        gpu_utilization = gpu_utilization.or(metrics.gpu_utilization);
                        gpu_memory_used = gpu_memory_used.or(metrics.gpu_memory_used);
                        gpu_memory_total = gpu_memory_total.or(metrics.gpu_memory_total);
                        gpu_fan_speed = gpu_fan_speed.or(metrics.gpu_fan_speed);
                        debug!("Collected AMD GPU metrics successfully");
                    }
                    Err(e) => {
                        debug!("Failed to collect AMD GPU metrics: {}", e);
                    }
                }
            } else {
                debug!("AMD monitor not available");
            }
        }

        // Collect Intel GPU metrics if available
        if config.gpu.enable_intel {
            match IntelMonitor::new() {
                Ok(monitor) if monitor.is_available() => match monitor.collect_metrics() {
                    Ok(metrics) => {
                        gpu_temperature = gpu_temperature.or(metrics.gpu_temperature);
                        gpu_power = gpu_power.or(metrics.gpu_power);
                        gpu_utilization = gpu_utilization.or(metrics.gpu_utilization);
                        gpu_memory_used = gpu_memory_used.or(metrics.gpu_memory_used);
                        gpu_memory_total = gpu_memory_total.or(metrics.gpu_memory_total);
                        gpu_fan_speed = gpu_fan_speed.or(metrics.gpu_fan_speed);
                        debug!("Collected Intel GPU metrics successfully");
                    }
                    Err(e) => {
                        debug!("Failed to collect Intel GPU metrics: {}", e);
                    }
                },
                Ok(_) => {
                    debug!("Intel GPU monitor not available");
                }
                Err(e) => {
                    debug!("Failed to initialize Intel monitor: {}", e);
                }
            }
        }

        Ok((
            gpu_temperature,
            gpu_power,
            gpu_utilization,
            gpu_memory_used,
            gpu_memory_total,
            gpu_fan_speed,
        ))
    }

    /// Collect CPU metrics using sysinfo
    async fn collect_cpu_metrics(
        config: &HardwareConfig,
    ) -> Result<(Option<f32>, Option<f32>), HardwareMonitorError> {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_cpu();

        let cpu_utilization = if config.cpu.monitor_all_cores {
            let mut total_usage = 0.0;
            let mut count = 0;

            for cpu in sys.cpus() {
                total_usage += cpu.cpu_usage();
                count += 1;

                if count >= config.cpu.max_cores {
                    break;
                }
            }

            if count > 0 {
                Some(total_usage / count as f32)
            } else {
                None
            }
        } else {
            sys.cpus().first().map(|cpu| cpu.cpu_usage())
        };

        // CPU temperature from hwmon sensors (Linux)
        let cpu_temperature = if config.cpu.enable_per_core_temp {
            Self::read_cpu_temperature_hwmon().ok()
        } else {
            None
        };

        Ok((cpu_temperature, cpu_utilization))
    }

    /// Read CPU temperature from hwmon sensors (Linux)
    fn read_cpu_temperature_hwmon() -> Result<f32, HardwareMonitorError> {
        // Try to read from common CPU temperature sensor paths
        let sensor_paths = vec![
            "/sys/class/hwmon/hwmon0/temp1_input",
            "/sys/class/hwmon/hwmon1/temp1_input",
            "/sys/class/hwmon/hwmon2/temp1_input",
            "/sys/class/thermal/thermal_zone0/temp",
        ];

        for path in sensor_paths {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(temp_millidegrees) = content.trim().parse::<u32>() {
                    // Convert millidegrees to celsius
                    return Ok(temp_millidegrees as f32 / 1000.0);
                }
            }
        }

        Err(HardwareMonitorError::CpuMonitoring(
            "No CPU temperature sensors found".to_string(),
        ))
    }

    /// Collect memory metrics using sysinfo
    async fn collect_memory_metrics(
        config: &HardwareConfig,
    ) -> Result<(Option<u64>, Option<u64>), HardwareMonitorError> {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_memory();

        let system_memory_used = if config.memory.monitor_system_ram {
            Some(sys.used_memory())
        } else {
            None
        };

        let system_memory_total = if config.memory.monitor_system_ram {
            Some(sys.total_memory())
        } else {
            None
        };

        Ok((system_memory_used, system_memory_total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silicon_synapse::config::HardwareConfig;
    use crate::silicon_synapse::telemetry_bus::TelemetryBus;
    use mockall::mock;

    // Mock NVML interface for testing
    mock! {
        NvidiaMonitor {}

        impl HardwareMonitor for NvidiaMonitor {
            fn collect_metrics(&self) -> Result<HardwareMetrics, HardwareMonitorError>;
            fn is_available(&self) -> bool;
            fn name(&self) -> &str;
        }
    }

    #[tokio::test]
    async fn test_hardware_collector_creation() {
        let config = HardwareConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
        let telemetry_sender = telemetry_bus.sender();

        let collector = HardwareCollector::new(config, telemetry_sender);
        assert!(collector.is_ok());
    }

    #[tokio::test]
    async fn test_hardware_collector_start_stop() {
        let config = HardwareConfig::default();
        let telemetry_config = crate::silicon_synapse::config::TelemetryConfig::default();
        let telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
        let telemetry_sender = telemetry_bus.sender();

        let mut collector = HardwareCollector::new(config, telemetry_sender).unwrap();

        assert!(collector.start().await.is_ok());
        assert!(collector.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_hardware_metrics_collection() {
        let config = HardwareConfig::default();
        let metrics = HardwareCollector::collect_hardware_metrics(&config, &None, &None).await;
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert!(metrics.timestamp.elapsed().as_secs() < 1);
    }

    #[tokio::test]
    async fn test_nvidia_monitor_creation() {
        // Test NVIDIA monitor creation (will fail gracefully if NVML not available)
        let monitor = NvidiaMonitor::new();
        assert!(monitor.is_ok());
    }

    #[tokio::test]
    async fn test_nvidia_monitor_availability() {
        let monitor = NvidiaMonitor::new().unwrap();
        // Availability depends on system - just test that the method exists
        let _available = monitor.is_available();
        assert_eq!(monitor.name(), "NVIDIA NVML");
    }

    #[tokio::test]
    async fn test_hardware_metrics_structure() {
        let metrics = HardwareMetrics {
            timestamp: Instant::now(),
            gpu_temperature: Some(45.0),
            gpu_power: Some(150.0),
            gpu_utilization: Some(75.0),
            gpu_memory_used: Some(2048),
            gpu_memory_total: Some(8192),
            gpu_fan_speed: Some(60.0),
            cpu_temperature: Some(55.0),
            cpu_utilization: Some(25.0),
            system_memory_used: Some(4096),
            system_memory_total: Some(16384),
        };

        assert!(metrics.gpu_temperature.is_some());
        assert!(metrics.gpu_power.is_some());
        assert!(metrics.gpu_utilization.is_some());
        assert!(metrics.gpu_memory_used.is_some());
        assert!(metrics.gpu_memory_total.is_some());
        assert!(metrics.gpu_fan_speed.is_some());
        assert!(metrics.cpu_temperature.is_some());
        assert!(metrics.cpu_utilization.is_some());
        assert!(metrics.system_memory_used.is_some());
        assert!(metrics.system_memory_total.is_some());
    }

    #[tokio::test]
    async fn test_cpu_metrics_collection() {
        let config = HardwareConfig::default();
        let result = HardwareCollector::collect_cpu_metrics(&config).await;
        assert!(result.is_ok());

        let (cpu_temp, cpu_util) = result.unwrap();
        // CPU utilization should be available, temperature might not be
        assert!(cpu_util.is_some() || cpu_temp.is_some());
    }

    #[tokio::test]
    async fn test_memory_metrics_collection() {
        let config = HardwareConfig::default();
        let result = HardwareCollector::collect_memory_metrics(&config).await;
        assert!(result.is_ok());

        let (used, total) = result.unwrap();
        assert!(used.is_some());
        assert!(total.is_some());

        if let (Some(u), Some(t)) = (used, total) {
            assert!(u <= t); // Used memory should not exceed total
        }
    }

    #[tokio::test]
    async fn test_gpu_metrics_collection_without_nvidia() {
        let config = HardwareConfig::default();
        let result = HardwareCollector::collect_gpu_metrics(&config, &None, &None).await;
        assert!(result.is_ok());

        let (temp, power, util, mem_used, mem_total, fan) = result.unwrap();
        // Without NVIDIA monitor, all values should be None
        assert!(temp.is_none());
        assert!(power.is_none());
        assert!(util.is_none());
        assert!(mem_used.is_none());
        assert!(mem_total.is_none());
        assert!(fan.is_none());
    }

    #[tokio::test]
    async fn test_performance_overhead() {
        let config = HardwareConfig {
            collection_interval_ms: 100, // Fast collection for testing
            ..HardwareConfig::default()
        };

        let start = Instant::now();
        let metrics = HardwareCollector::collect_hardware_metrics(&config, &None, &None).await;
        let duration = start.elapsed();

        assert!(metrics.is_ok());
        // Should complete in less than 100ms (performance requirement)
        assert!(duration.as_millis() < 100);
    }
}

#[async_trait]
impl Collector for HardwareCollector {
    fn id(&self) -> &str {
        "hardware"
    }

    fn name(&self) -> &str {
        "Hardware Monitor"
    }

    fn config(&self) -> &CollectorConfig {
        &self.collector_config
    }

    async fn initialize(&mut self) -> CollectorResult<()> {
        // Hardware collector initialization is done in new()
        tracing::info!("Hardware collector initialized");
        Ok(())
    }

    async fn start(&mut self) -> CollectorResult<()> {
        self.start_internal().await.map_err(|e| {
            CollectorError::InitializationFailed(format!(
                "Failed to start hardware collector: {}",
                e
            ))
        })
    }

    async fn stop(&mut self) -> CollectorResult<()> {
        self.stop_internal().await.map_err(|e| {
            CollectorError::ShutdownFailed(format!("Failed to stop hardware collector: {}", e))
        })
    }

    async fn collect(&self) -> CollectorResult<CollectedMetrics> {
        let start_time = std::time::Instant::now();

        // Collect hardware metrics
        let hardware_metrics =
            Self::collect_hardware_metrics(&self.config, &self.nvidia_monitor, &self.amd_monitor)
                .await?;

        let collection_duration = start_time.elapsed();

        // Convert to CollectedMetrics format
        let mut metrics_map = std::collections::HashMap::new();

        // Add hardware metrics to the map
        if let Some(temp) = hardware_metrics.gpu_temperature {
            metrics_map.insert(
                "gpu_temperature".to_string(),
                MetricValue::Float(temp as f64),
            );
        }
        if let Some(power) = hardware_metrics.gpu_power {
            metrics_map.insert("gpu_power".to_string(), MetricValue::Float(power as f64));
        }
        if let Some(vram) = hardware_metrics.gpu_memory_used {
            metrics_map.insert(
                "gpu_memory_usage".to_string(),
                MetricValue::Float(vram as f64),
            );
        }

        Ok(CollectedMetrics {
            collector_id: self.id().to_string(),
            timestamp: std::time::SystemTime::now(),
            collection_duration,
            metrics: metrics_map,
            metadata: std::collections::HashMap::new(),
        })
    }

    fn is_running(&self) -> bool {
        *self.is_running.blocking_read()
    }

    async fn health_check(&self) -> CollectorResult<CollectorHealth> {
        // Basic health check
        Ok(CollectorHealth {
            is_healthy: true,
            last_collection: Some(std::time::SystemTime::now()),
            failure_count: 0,
            error_message: None,
            stats: CollectorStats::default(),
        })
    }

    async fn shutdown(&mut self) -> CollectorResult<()> {
        // Shutdown hardware monitoring
        Ok(())
    }

    async fn update_config(&mut self, config: CollectorConfig) -> CollectorResult<()> {
        // Update hardware collector configuration
        Ok(())
    }
}
