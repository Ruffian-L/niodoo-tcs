use std::time::{Duration, Instant};
use std::io;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use ratatui::{
    backend::CrosstermBackend,
    crossterm::{
        event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
};

#[derive(Clone)]
struct DashboardState {
    current_prompt: String,
    current_response: String,
    iteration: u64,
    total_iterations: u64,
    success_count: u64,
    failure_count: u64,
    avg_latency_ms: f64,
    avg_rouge: f64,
    throughput_ops_per_sec: f64,
    recent_errors: Vec<(String, String)>, // (time, error_msg)
    recent_prompts: Vec<String>,
    qlora_trainings: u64,
    memory_upserts: u64,
    grpc_failures: u64,
    status: String,
    last_update: Instant,
    progress_percent: f64,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            current_prompt: String::new(),
            current_response: String::new(),
            iteration: 0,
            total_iterations: 0,
            success_count: 0,
            failure_count: 0,
            avg_latency_ms: 0.0,
            avg_rouge: 0.0,
            throughput_ops_per_sec: 0.0,
            recent_errors: Vec::new(),
            recent_prompts: Vec::new(),
            qlora_trainings: 0,
            memory_upserts: 0,
            grpc_failures: 0,
            status: "Starting...".to_string(),
            last_update: Instant::now(),
            progress_percent: 0.0,
        }
    }
}

impl DashboardState {
    fn progress_percent(&self) -> f64 {
        self.progress_percent
    }
}

#[tokio::main]
async fn main() -> io::Result<()> {
    // Get log file path from args or use default
    let log_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/soak_100cycles_50prompts.log".to_string());

    let state = Arc::new(RwLock::new(DashboardState {
        status: "Monitoring...".to_string(),
        last_update: Instant::now(),
        ..Default::default()
    }));

    // Spawn background task to monitor logs
    let state_clone = state.clone();
    let log_file_clone = log_file.clone();
    tokio::spawn(async move {
        monitor_logs(state_clone, log_file_clone).await;
    });

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut should_quit = false;
    while !should_quit {
        terminal.draw(|f| ui(f, &state))?;

        if crossterm::event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => should_quit = true,
                    _ => {}
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

async fn monitor_logs(state: Arc<RwLock<DashboardState>>, log_file: String) {
    // Try to open log file and tail it
    loop {
        match File::open(&log_file).await {
            Ok(file) => {
                let reader = BufReader::new(file);
                let mut lines = reader.lines();
                
                // Skip to end if file exists
                let mut line_count = 0;
                while let Ok(Some(_)) = lines.next_line().await {
                    line_count += 1;
                    if line_count > 1000 {
                        break; // Skip old logs
                    }
                }
                
                // Now read new lines
                while let Ok(Some(line)) = lines.next_line().await {
                    parse_log_line(&state, &line).await;
                }
            }
            Err(_) => {
                // File doesn't exist yet, wait and retry
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        }
        
        // File closed or EOF, wait and retry
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn parse_log_line(state: &Arc<RwLock<DashboardState>>, line: &str) {
    let mut state_guard = state.write().await;
    state_guard.last_update = Instant::now();
    
    // Parse iteration
    if line.contains("Iteration") || line.contains("iteration") {
        if let Some(iter_str) = extract_number_after(line, "iteration") {
            if let Ok(iter) = iter_str.parse::<u64>() {
                state_guard.iteration = iter;
            }
        }
    }
    
    // Parse prompt
    if line.contains("Processing prompt") || line.contains("prompt") {
        if let Some(prompt) = extract_prompt(line) {
            state_guard.current_prompt = prompt.clone();
            state_guard.recent_prompts.push(prompt);
            if state_guard.recent_prompts.len() > 5 {
                state_guard.recent_prompts.remove(0);
            }
        }
    }
    
    // Parse latency
    if line.contains("latency_ms=") {
        if let Some(latency) = extract_number_after(line, "latency_ms=") {
            if let Ok(lat_ms) = latency.parse::<f64>() {
                state_guard.avg_latency_ms = lat_ms;
                state_guard.success_count += 1;
            }
        }
    }
    
    // Parse ROUGE
    if line.contains("rouge=") || line.contains("ROUGE=") {
        if let Some(rouge_str) = extract_number_after(line, "rouge=") {
            if let Ok(rouge) = rouge_str.parse::<f64>() {
                state_guard.avg_rouge = rouge;
            }
        }
    }
    
    // Parse errors
    if line.contains("ERROR") || line.contains("WARN") || line.contains("failed") {
        let time = chrono::Local::now().format("%H:%M:%S").to_string();
        let error_msg = if line.len() > 100 {
            format!("{}...", &line[..100])
        } else {
            line.to_string()
        };
        state_guard.recent_errors.push((time, error_msg));
        if state_guard.recent_errors.len() > 10 {
            state_guard.recent_errors.remove(0);
        }
        state_guard.failure_count += 1;
    }
    
    // Parse QLoRA training
    if line.contains("QLoRA") || line.contains("LoRA training") {
        state_guard.qlora_trainings += 1;
    }
    
    // Parse memory upserts
    if line.contains("stored ERAG memory") || line.contains("upsert") {
        state_guard.memory_upserts += 1;
    }
    
    // Parse gRPC failures
    if line.contains("gRPC") && (line.contains("failed") || line.contains("fallback")) {
        state_guard.grpc_failures += 1;
    }
    
    // Parse progress
    if line.contains("Progress:") {
        if let Some(percent_str) = extract_number_after(line, "(") {
            if let Ok(percent) = percent_str.replace("%", "").parse::<f64>() {
                state_guard.progress_percent = percent;
            }
        }
    }
    
    // Calculate throughput
    if state_guard.last_update.elapsed().as_secs() > 0 {
        let elapsed = state_guard.last_update.elapsed().as_secs_f64();
        state_guard.throughput_ops_per_sec = state_guard.success_count as f64 / elapsed.max(1.0);
    }
    
    drop(state_guard);
}

fn extract_number_after(text: &str, prefix: &str) -> Option<String> {
    if let Some(pos) = text.find(prefix) {
        let after = &text[pos + prefix.len()..];
        let number: String = after
            .chars()
            .take_while(|c| c.is_numeric() || *c == '.' || *c == '-')
            .collect();
        if !number.is_empty() {
            return Some(number);
        }
    }
    None
}

fn extract_prompt(text: &str) -> Option<String> {
    if let Some(pos) = text.find("prompt") {
        let after = &text[pos..];
        // Try to extract quoted string or text after colon
        if let Some(start) = after.find('"') {
            if let Some(end) = after[start+1..].find('"') {
                return Some(after[start+1..start+1+end].to_string());
            }
        }
        if let Some(start) = after.find(':') {
            let prompt = after[start+1..].trim();
            if prompt.len() > 0 && prompt.len() < 100 {
                return Some(prompt.to_string());
            }
        }
    }
    None
}

fn ui(f: &mut Frame, state: &Arc<RwLock<DashboardState>>) {
    let state_guard = tokio::runtime::Handle::current().block_on(state.read());
    
    let size = f.size();
    
    // Create main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(6),  // Metrics row
            Constraint::Length(6),  // Current operation
            Constraint::Min(8),     // Recent activity
            Constraint::Length(3),  // Footer
        ])
        .split(size);

    // Header
    let header_text = format!("🚀 NIODOO-TCS LIVE DASHBOARD | Status: {}", state_guard.status);
    let header = Paragraph::new(header_text)
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // Metrics row - 4 columns
    let metrics_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[1]);

    // Progress gauge
    let progress = state_guard.progress_percent();
    let progress_text = format!("{:.1}%", progress);
    let progress_gauge = Gauge::default()
        .block(Block::default().title("📊 Progress").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Green))
        .percent(progress.min(100.0) as u16)
        .label(progress_text);
    f.render_widget(progress_gauge, metrics_chunks[0]);

    // Latency
    let latency_text = format!("{:.0}ms", state_guard.avg_latency_ms);
    let latency_color = if state_guard.avg_latency_ms < 3000.0 {
        Color::Green
    } else if state_guard.avg_latency_ms < 5000.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let latency_widget = Paragraph::new(latency_text)
        .block(Block::default().title("⚡ Latency").borders(Borders::ALL))
        .style(Style::default().fg(latency_color));
    f.render_widget(latency_widget, metrics_chunks[1]);

    // ROUGE score
    let rouge_text = format!("{:.3}", state_guard.avg_rouge);
    let rouge_color = if state_guard.avg_rouge > 0.4 {
        Color::Green
    } else if state_guard.avg_rouge > 0.3 {
        Color::Yellow
    } else {
        Color::Red
    };
    let rouge_widget = Paragraph::new(rouge_text)
        .block(Block::default().title("✨ Quality (ROUGE)").borders(Borders::ALL))
        .style(Style::default().fg(rouge_color));
    f.render_widget(rouge_widget, metrics_chunks[2]);

    // Throughput
    let throughput_text = format!("{:.2} ops/s", state_guard.throughput_ops_per_sec);
    let throughput_widget = Paragraph::new(throughput_text)
        .block(Block::default().title("🏃 Throughput").borders(Borders::ALL))
        .style(Style::default().fg(Color::Blue));
    f.render_widget(throughput_widget, metrics_chunks[3]);

    // Current prompt
    let prompt_text = if state_guard.current_prompt.is_empty() {
        "Waiting for prompts...".to_string()
    } else {
        format!("📝 {}", state_guard.current_prompt)
    };
    let prompt_para = Paragraph::new(prompt_text)
        .block(Block::default().title("Current Prompt").borders(Borders::ALL))
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: true });
    f.render_widget(prompt_para, chunks[2]);

    // Recent activity - split into errors and recent prompts
    let activity_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[3]);

    // Errors table
    let error_rows: Vec<Row> = state_guard
        .recent_errors
        .iter()
        .rev()
        .take(8)
        .map(|(time, msg)| {
            Row::new(vec![
                time.clone(),
                if msg.len() > 40 { format!("{}...", &msg[..40]) } else { msg.clone() },
            ])
        })
        .collect();

    let error_table = Table::new(error_rows, &[Constraint::Length(12), Constraint::Min(30)])
        .block(Block::default().title("🚨 Recent Errors").borders(Borders::ALL))
        .style(Style::default().fg(Color::Red));
    f.render_widget(error_table, activity_chunks[0]);

    // Recent prompts
    let prompt_items: Vec<ListItem> = state_guard
        .recent_prompts
        .iter()
        .rev()
        .take(8)
        .map(|p| {
            let text = if p.len() > 50 {
                format!("{}...", &p[..50])
            } else {
                p.clone()
            };
            ListItem::new(text)
        })
        .collect();

    let prompt_list = List::new(prompt_items)
        .block(Block::default().title("📋 Recent Prompts").borders(Borders::ALL))
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(prompt_list, activity_chunks[1]);

    // Footer with stats
    let footer_text = format!(
        "Iteration: {}/{} | ✅ Success: {} | ❌ Failures: {} | 🧠 LoRA: {} | 💾 Memory: {} | gRPC Failures: {} | Press 'q' to quit",
        state_guard.iteration,
        state_guard.total_iterations,
        state_guard.success_count,
        state_guard.failure_count,
        state_guard.qlora_trainings,
        state_guard.memory_upserts,
        state_guard.grpc_failures
    );
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::Cyan))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[4]);
}
