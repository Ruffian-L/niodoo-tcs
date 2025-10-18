use niodoo_consciousness::silicon_synapse::{Config, SiliconSynapse};
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};
use sysinfo::{Pid, System};
use tokio::runtime::Runtime;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let config = Config::load("config/silicon_synapse.toml").unwrap_or_default();
        let mut synapse = SiliconSynapse::new(config).await.unwrap();
        synapse.start().await.unwrap();

        let mut sys = System::new_all();
        let pid = Pid::from_u32(std::process::id());
        let mut file = BufWriter::new(File::create("./logs/benchmark_results.jsonl").unwrap());

        let num_iterations = 100;
        let start_total = Instant::now();
        let mut peak_memory_mb = 0.0;

        for i in 0..num_iterations {
            let start = Instant::now();

            // Dummy consciousness update - replace with actual call
            // For example: let _ = consciousness.update_state(dummy_state).await;
            tokio::time::sleep(Duration::from_millis(10)).await; // Simulate work

            let latency = start.elapsed().as_millis() as f64;

            sys.refresh_process(pid);
            let memory_mb = sys
                .process(pid)
                .map(|p| p.memory() as f64 / 1_048_576.0)
                .unwrap_or(0.0);

            // Track peak memory usage
            if memory_mb > peak_memory_mb {
                peak_memory_mb = memory_mb;
            }

            let entry = json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "iteration": i,
                "latency_ms": latency,
                "memory_mb": memory_mb,
                "cumulative_ups": (i as f64 + 1.0) / start_total.elapsed().as_secs_f64(),
            });

            writeln!(file, "{}", entry).unwrap();
            file.flush().unwrap();

            info!(
                "Iteration {}: latency {}ms, memory {}MB",
                i, latency, memory_mb
            );
        }

        let total_time = start_total.elapsed();
        let avg_latency = total_time / num_iterations as u32;
        let ups = num_iterations as f64 / total_time.as_secs_f64();

        info!(
            "Benchmark complete: avg latency {}ms/update, peak mem {:.1}MB, UPS {:.1}",
            avg_latency.as_millis(),
            peak_memory_mb,
            ups
        );

        // Assert targets
        assert!(
            avg_latency.as_millis() < 20,
            "Average latency exceeded 20ms"
        );
        assert!(peak_memory_mb < 4096.0, "Memory exceeded 4GB");
        assert!(ups > 100.0, "UPS below 100");

        synapse.shutdown().await.unwrap();

        Ok(())
    })
}
