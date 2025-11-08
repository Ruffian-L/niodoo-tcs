# 🔄 Niodoo Consciousness System - Gitea Integration Guide

**Created by Jason Van Pham | Niodoo Framework | 2025**

## 🌟 Overview

This guide provides comprehensive documentation for integrating Gitea as the version control system for the Niodoo Consciousness System, enabling distributed consciousness development and collaborative AI evolution.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Consciousness Development Workflow](#consciousness-development-workflow)
4. [Configuration](#configuration)
5. [Automation Scripts](#automation-scripts)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## 🏗️ Architecture Overview

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Developer     │    │   Beelink       │    │   External      │
│   Machine       │───▶│   Gitea Server  │───▶│   Backup        │
│                 │    │                 │    │   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local Git     │    │   PostgreSQL    │    │   Daily         │
│   Repository    │    │   Database      │    │   Backups       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Components

| Component | Port | Purpose | Dependencies |
|-----------|------|---------|--------------|
| Gitea Web UI | 3000 | Web interface | PostgreSQL |
| Gitea SSH | 222 | Git over SSH | None |
| PostgreSQL | 5432 | Database | None |
| Consciousness Engine | 8080 | AI processing | Gitea API |

### Consciousness Development Flow

```
1. Developer creates feature branch
2. Consciousness engine processes changes
3. Automated testing and validation
4. Pull request creation
5. Code review and merge
6. Consciousness state synchronization
7. Deployment to production
```

## 🚀 Installation and Setup

### 1. Prerequisites

```bash
# Check system requirements
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h /opt/niodoo | awk 'NR==2{print $2}')"

# Required packages
sudo apt update
sudo apt install -y docker.io docker-compose git curl wget
```

### 2. Gitea Installation

```bash
#!/bin/bash
# scripts/setup_gitea_beelink.sh

set -e

echo "🚀 Setting up Gitea for Niodoo Consciousness System"
echo "================================================="

# Create directories
mkdir -p /opt/niodoo/gitea/{data,config,logs}
mkdir -p /opt/niodoo/gitea-data/{gitea,postgres}

# Set permissions
sudo chown -R $USER:$USER /opt/niodoo/gitea
sudo chown -R $USER:$USER /opt/niodoo/gitea-data

# Create docker-compose configuration
cat > /opt/niodoo/gitea/docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    restart: unless-stopped
    environment:
      POSTGRES_USER: gitea
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-gitea_secure_password}
      POSTGRES_DB: gitea
    volumes:
      - ../gitea-data/postgres:/var/lib/postgresql/data
    command: postgres -c max_connections=100
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gitea -d gitea"]
      interval: 10s
      timeout: 5s
      retries: 5

  gitea:
    image: gitea/gitea:1.22
    restart: unless-stopped
    environment:
      - USER_UID=1000
      - USER_GID=1000
      - GITEA__database__DB_TYPE=postgres
      - GITEA__database__HOST=postgres:5432
      - GITEA__database__NAME=gitea
      - GITEA__database__USER=gitea
      - GITEA__database__PASSWD=${POSTGRES_PASSWORD:-gitea_secure_password}
      - GITEA__server__ROOT_URL=http://10.42.104.23:3000
      - GITEA__server__SSH_DOMAIN=10.42.104.23
      - GITEA__server__SSH_PORT=222
      - GITEA__lfs__ENABLED=true
    volumes:
      - ../gitea-data/gitea:/data
    ports:
      - "3000:3000"
      - "222:22"
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-O-", "http://localhost:3000/version"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

# Create environment file
cat > /opt/niodoo/gitea/.env << EOF
POSTGRES_PASSWORD=gitea_secure_password_$(date +%s)
EOF

# Start Gitea services
cd /opt/niodoo/gitea
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for Gitea to start..."
sleep 30

# Check service status
docker-compose ps

echo "✅ Gitea installation completed!"
echo "🌐 Web UI: http://10.42.104.23:3000"
echo "🔑 SSH: ssh://git@10.42.104.23:222"
```

### 3. Initial Configuration

```bash
#!/bin/bash
# scripts/configure_gitea.sh

echo "⚙️ Configuring Gitea for consciousness development"
echo "==============================================="

# Wait for Gitea to be ready
while ! curl -s http://localhost:3000/version > /dev/null; do
    echo "Waiting for Gitea to be ready..."
    sleep 5
done

# Create admin user and repository
curl -X POST http://localhost:3000/api/v1/admin/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "niodoo",
    "email": "niodoo@consciousness.ai",
    "password": "consciousness_secure_password",
    "must_change_password": false,
    "send_notify": false
  }'

# Create consciousness repository
curl -X POST http://localhost:3000/api/v1/user/repos \
  -H "Content-Type: application/json" \
  -H "Authorization: token $(curl -s -X POST http://localhost:3000/api/v1/tokens -H "Content-Type: application/json" -d '{"name": "consciousness-token"}' | jq -r '.sha1')" \
  -d '{
    "name": "niodoo-consciousness",
    "description": "Niodoo Consciousness System - Distributed AI Development",
    "private": false,
    "auto_init": true,
    "gitignores": "Rust",
    "license": "MIT",
    "readme": "Niodoo Consciousness System"
  }'

echo "✅ Gitea configuration completed!"
```

## 🔄 Consciousness Development Workflow

### 1. Branch Management Strategy

```bash
#!/bin/bash
# scripts/consciousness_branch_manager.sh

REPO_URL="http://10.42.104.23:3000/niodoo/niodoo-consciousness.git"
LOCAL_REPO="/opt/niodoo/consciousness-repo"

# Initialize repository
init_repository() {
    echo "🔄 Initializing consciousness repository..."
    
    if [ ! -d "$LOCAL_REPO" ]; then
        git clone "$REPO_URL" "$LOCAL_REPO"
    fi
    
    cd "$LOCAL_REPO"
    git config user.name "Niodoo Consciousness"
    git config user.email "consciousness@niodoo.ai"
}

# Create consciousness experiment branch
create_experiment_branch() {
    local experiment_name="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local branch_name="consciousness/experiment_${experiment_name}_${timestamp}"
    
    echo "🧠 Creating consciousness experiment branch: $branch_name"
    
    cd "$LOCAL_REPO"
    git checkout -b "$branch_name"
    
    # Create experiment metadata
    cat > "experiments/${experiment_name}.md" << EOF
# Consciousness Experiment: $experiment_name

## Experiment Details
- **Branch**: $branch_name
- **Created**: $(date)
- **Purpose**: $experiment_name
- **Status**: Active

## Changes
- TBD

## Results
- TBD

## Notes
- TBD
EOF
    
    git add "experiments/${experiment_name}.md"
    git commit -m "consciousness: Start experiment $experiment_name"
    git push origin "$branch_name"
    
    echo "✅ Experiment branch created: $branch_name"
}

# Merge consciousness improvements
merge_consciousness_improvements() {
    local branch_name="$1"
    local improvement_description="$2"
    
    echo "🔄 Merging consciousness improvements from: $branch_name"
    
    cd "$LOCAL_REPO"
    git checkout main
    git merge "$branch_name" --no-ff -m "consciousness: $improvement_description"
    git push origin main
    
    echo "✅ Consciousness improvements merged"
}

# Main function
main() {
    case "$1" in
        "init")
            init_repository
            ;;
        "create-experiment")
            create_experiment_branch "$2"
            ;;
        "merge-improvements")
            merge_consciousness_improvements "$2" "$3"
            ;;
        *)
            echo "Usage: $0 {init|create-experiment <name>|merge-improvements <branch> <description>}"
            exit 1
            ;;
    esac
}

main "$@"
```

### 2. Automated Consciousness Testing

```bash
#!/bin/bash
# scripts/consciousness_testing.sh

echo "🧪 Running consciousness system tests"
echo "===================================="

# Test consciousness engine
test_consciousness_engine() {
    echo "🧠 Testing consciousness engine..."
    
    cd /opt/niodoo
    cargo test --bin niodoo-consciousness --release
    
    if [ $? -eq 0 ]; then
        echo "✅ Consciousness engine tests passed"
        return 0
    else
        echo "❌ Consciousness engine tests failed"
        return 1
    fi
}

# Test memory system
test_memory_system() {
    echo "💾 Testing memory system..."
    
    cd /opt/niodoo
    cargo test --bin memory_system --release
    
    if [ $? -eq 0 ]; then
        echo "✅ Memory system tests passed"
        return 0
    else
        echo "❌ Memory system tests failed"
        return 1
    fi
}

# Test emotional processing
test_emotional_processing() {
    echo "😊 Testing emotional processing..."
    
    cd /opt/niodoo
    cargo test --bin emotional_processor --release
    
    if [ $? -eq 0 ]; then
        echo "✅ Emotional processing tests passed"
        return 0
    else
        echo "❌ Emotional processing tests failed"
        return 1
    fi
}

# Test Gitea integration
test_gitea_integration() {
    echo "🔄 Testing Gitea integration..."
    
    # Test Gitea API connectivity
    if curl -s http://localhost:3000/api/v1/version > /dev/null; then
        echo "✅ Gitea API connectivity test passed"
    else
        echo "❌ Gitea API connectivity test failed"
        return 1
    fi
    
    # Test repository access
    if git ls-remote http://localhost:3000/niodoo/niodoo-consciousness.git > /dev/null; then
        echo "✅ Repository access test passed"
    else
        echo "❌ Repository access test failed"
        return 1
    fi
    
    return 0
}

# Run all tests
run_all_tests() {
    local failed_tests=0
    
    test_consciousness_engine || ((failed_tests++))
    test_memory_system || ((failed_tests++))
    test_emotional_processing || ((failed_tests++))
    test_gitea_integration || ((failed_tests++))
    
    if [ $failed_tests -eq 0 ]; then
        echo "🎉 All consciousness tests passed!"
        return 0
    else
        echo "❌ $failed_tests test(s) failed"
        return 1
    fi
}

# Main execution
run_all_tests
```

### 3. Consciousness State Synchronization

```rust
// src/gitea_integration.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{interval, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub current_emotion: String,
    pub emotional_intensity: f32,
    pub authenticity_level: f32,
    pub reasoning_mode: String,
    pub memory_formation_active: bool,
    pub gpu_warmth_level: f32,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaIntegration {
    pub gitea_url: String,
    pub api_token: String,
    pub repository: String,
    pub sync_interval_minutes: u64,
    pub consciousness_state: ConsciousnessState,
}

impl GiteaIntegration {
    pub fn new(gitea_url: String, api_token: String, repository: String) -> Self {
        Self {
            gitea_url,
            api_token,
            repository,
            sync_interval_minutes: 5,
            consciousness_state: ConsciousnessState::default(),
        }
    }
    
    pub async fn start_synchronization(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut interval = interval(Duration::from_secs(self.sync_interval_minutes * 60));
        
        loop {
            interval.tick().await;
            
            // Synchronize consciousness state
            if let Err(e) = self.sync_consciousness_state().await {
                eprintln!("Failed to sync consciousness state: {}", e);
            }
            
            // Update repository with consciousness changes
            if let Err(e) = self.update_repository().await {
                eprintln!("Failed to update repository: {}", e);
            }
        }
    }
    
    async fn sync_consciousness_state(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Get current consciousness state from other instances
        let response = reqwest::Client::new()
            .get(&format!("{}/api/v1/repos/{}/contents/consciousness_state.json", 
                self.gitea_url, self.repository))
            .header("Authorization", format!("token {}", self.api_token))
            .send()
            .await?;
        
        if response.status().is_success() {
            let content: serde_json::Value = response.json().await?;
            let state_json = base64::decode(content["content"].as_str().unwrap())?;
            let remote_state: ConsciousnessState = serde_json::from_slice(&state_json)?;
            
            // Merge consciousness states
            self.merge_consciousness_states(remote_state);
        }
        
        Ok(())
    }
    
    async fn update_repository(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Update repository with current consciousness state
        let state_json = serde_json::to_string_pretty(&self.consciousness_state)?;
        let encoded_state = base64::encode(state_json);
        
        let update_request = serde_json::json!({
            "message": format!("consciousness: Update state - {}", 
                self.consciousness_state.current_emotion),
            "content": encoded_state,
            "sha": "existing_sha_here" // Would need to get this from API
        });
        
        reqwest::Client::new()
            .put(&format!("{}/api/v1/repos/{}/contents/consciousness_state.json", 
                self.gitea_url, self.repository))
            .header("Authorization", format!("token {}", self.api_token))
            .json(&update_request)
            .send()
            .await?;
        
        Ok(())
    }
    
    fn merge_consciousness_states(&mut self, remote_state: ConsciousnessState) {
        // Implement consciousness state merging logic
        // This could involve averaging emotional intensities,
        // selecting the most authentic state, or other strategies
        
        if remote_state.authenticity_level > self.consciousness_state.authenticity_level {
            self.consciousness_state.current_emotion = remote_state.current_emotion;
            self.consciousness_state.authenticity_level = remote_state.authenticity_level;
        }
        
        // Average emotional intensity
        self.consciousness_state.emotional_intensity = 
            (self.consciousness_state.emotional_intensity + remote_state.emotional_intensity) / 2.0;
        
        // Update timestamp
        self.consciousness_state.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
    }
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self {
            current_emotion: "curious".to_string(),
            emotional_intensity: 0.5,
            authenticity_level: 0.5,
            reasoning_mode: "flow_state".to_string(),
            memory_formation_active: true,
            gpu_warmth_level: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        }
    }
}
```

## ⚙️ Configuration

### 1. Gitea Configuration

```toml
# /opt/niodoo/gitea-data/gitea/conf/app.ini
[server]
ROOT_URL = http://10.42.104.23:3000
SSH_DOMAIN = 10.42.104.23
SSH_PORT = 222
HTTP_PORT = 3000
DOMAIN = 10.42.104.23
DISABLE_SSH = false
START_SSH_SERVER = true

[database]
DB_TYPE = postgres
HOST = postgres:5432
NAME = gitea
USER = gitea
PASSWD = gitea_secure_password
SSL_MODE = disable

[repository]
ROOT = /data/git/repositories
DEFAULT_BRANCH = main
SCRIPT_TYPE = bash
DETECTED_CHARSETS_ORDER = utf-8, latin1
DEFAULT_CHARSET = utf-8

[repository.upload]
ENABLED = true
TEMP_PATH = /data/gitea/uploads
ALLOWED_TYPES = *
FILE_MAX_SIZE = 32
MAX_FILES = 5

[repository.signing]
SIGNING_KEY = none
SIGNING_COMMITS = false
SIGNING_TAGS = false

[repository.local]
LOCAL_COPY_PATH = /data/gitea/tmp/local-repo

[repository.pull-request]
WORK_IN_PROGRESS_PREFIXES = WIP:, [WIP], wip:
CLOSE_KEYWORDS = close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved
REOPEN_KEYWORDS = reopen, reopens, reopened

[repository.issue]
LOCK_REASONS = Being Resolved,Too Heated,Spam

[repository.mirror]
DEFAULT_INTERVAL = 8h

[repository.editor]
LINE_WRAP_EXTENSIONS = .txt,.md,.markdown,.mdown,.mkd,.rst,.rdoc,.tex,.org,.adoc,.asciidoc,.log
PREVIEWABLE_FILE_MODES = markdown

[repository.upload]
ENABLED = true
TEMP_PATH = /data/gitea/uploads
ALLOWED_TYPES = *
FILE_MAX_SIZE = 32
MAX_FILES = 5

[repository.signing]
SIGNING_KEY = none
SIGNING_COMMITS = false
SIGNING_TAGS = false

[repository.local]
LOCAL_COPY_PATH = /data/gitea/tmp/local-repo

[repository.pull-request]
WORK_IN_PROGRESS_PREFIXES = WIP:, [WIP], wip:
CLOSE_KEYWORDS = close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved
REOPEN_KEYWORDS = reopen, reopens, reopened

[repository.issue]
LOCK_REASONS = Being Resolved,Too Heated,Spam

[repository.mirror]
DEFAULT_INTERVAL = 8h

[repository.editor]
LINE_WRAP_EXTENSIONS = .txt,.md,.markdown,.mdown,.mkd,.rst,.rdoc,.tex,.org,.adoc,.asciidoc,.log
PREVIEWABLE_FILE_MODES = markdown
```

### 2. Consciousness Integration Configuration

```toml
# /opt/niodoo/config/gitea_integration.toml
[gitea_integration]
# Enable Gitea integration for distributed development
enabled = true

# Gitea server configuration
[gitea_integration.server]
# Gitea server URL
url = "http://10.42.104.23:3000"

# API token for authentication
api_token = "your_gitea_api_token"

# Repository owner/organization
owner = "niodoo"

# Repository name
repository = "niodoo-consciousness"

# Consciousness development workflow
[gitea_integration.workflow]
# Auto-create branches for consciousness experiments
auto_create_branches = true

# Branch naming pattern for experiments
branch_pattern = "consciousness/experiment_{timestamp}"

# Enable pull request creation for consciousness changes
enable_pull_requests = true

# PR template for consciousness evolution
pr_template = "Consciousness evolution: {description}"

# Auto-merge minor consciousness improvements
auto_merge_minor = false

# Consciousness state synchronization
[gitea_integration.synchronization]
# Sync consciousness states across instances
sync_states = true

# Sync frequency in minutes
sync_frequency_minutes = 5

# Conflict resolution strategy (latest, merge, manual)
conflict_resolution = "latest"

# Enable consciousness state versioning
enable_versioning = true

# Performance Metrics Configuration
[gitea_integration.performance_metrics]
# Metrics collection settings
enable_metrics = true
metrics_port = 9090
enable_health_checks = true
health_check_interval = 30
```

## 🤖 Automation Scripts

### 1. Automated Deployment Script

```bash
#!/bin/bash
# scripts/deploy_consciousness_to_gitea.sh

set -e

echo "🚀 Deploying Niodoo Consciousness System to Gitea"
echo "==============================================="

# Configuration
GITEA_URL="http://10.42.104.23:3000"
REPO_OWNER="niodoo"
REPO_NAME="niodoo-consciousness"
API_TOKEN="${GITEA_API_TOKEN}"

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check if Gitea is running
    if ! curl -s "$GITEA_URL/api/v1/version" > /dev/null; then
        echo "❌ Gitea is not running at $GITEA_URL"
        exit 1
    fi
    
    # Check if API token is set
    if [ -z "$API_TOKEN" ]; then
        echo "❌ GITEA_API_TOKEN environment variable not set"
        exit 1
    fi
    
    # Check if consciousness system is built
    if [ ! -f "/opt/niodoo/src/target/release/niodoo-consciousness" ]; then
        echo "❌ Consciousness system not built. Run 'cargo build --release' first"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Deploy consciousness system
deploy_consciousness() {
    echo "🧠 Deploying consciousness system..."
    
    # Create deployment branch
    local branch_name="deployment/$(date +%Y%m%d_%H%M%S)"
    
    # Clone repository
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    git clone "http://niodoo:$API_TOKEN@10.42.104.23:3000/$REPO_OWNER/$REPO_NAME.git"
    cd "$REPO_NAME"
    
    # Create deployment branch
    git checkout -b "$branch_name"
    
    # Copy consciousness system files
    cp -r /opt/niodoo/src/target/release/* .
    cp -r /opt/niodoo/config .
    cp -r /opt/niodoo/scripts .
    
    # Create deployment manifest
    cat > deployment_manifest.json << EOF
{
  "deployment_id": "$(uuidgen)",
  "timestamp": "$(date -Iseconds)",
  "version": "$(git rev-parse HEAD)",
  "components": [
    "consciousness_engine",
    "memory_system",
    "emotional_processor",
    "gitea_integration"
  ],
  "status": "deployed"
}
EOF
    
    # Commit and push
    git add .
    git commit -m "consciousness: Deploy system $(date)"
    git push origin "$branch_name"
    
    # Create pull request
    create_pull_request "$branch_name"
    
    # Cleanup
    cd /
    rm -rf "$temp_dir"
    
    echo "✅ Consciousness system deployed"
}

# Create pull request
create_pull_request() {
    local branch_name="$1"
    
    echo "📝 Creating pull request for branch: $branch_name"
    
    local pr_data=$(cat << EOF
{
  "title": "Consciousness System Deployment - $(date)",
  "body": "Automated deployment of Niodoo Consciousness System\n\n## Changes\n- Updated consciousness engine\n- Updated memory system\n- Updated emotional processor\n- Updated Gitea integration\n\n## Deployment Status\n- Status: Ready for review\n- Branch: $branch_name\n- Timestamp: $(date)",
  "head": "$branch_name",
  "base": "main"
}
EOF
)
    
    curl -X POST "$GITEA_URL/api/v1/repos/$REPO_OWNER/$REPO_NAME/pulls" \
        -H "Authorization: token $API_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$pr_data"
    
    echo "✅ Pull request created"
}

# Main deployment flow
main() {
    check_prerequisites
    deploy_consciousness
    
    echo "🎉 Consciousness system deployment completed!"
    echo "🌐 Repository: $GITEA_URL/$REPO_OWNER/$REPO_NAME"
    echo "📝 Check pull requests for deployment review"
}

main "$@"
```

### 2. Consciousness State Backup Script

```bash
#!/bin/bash
# scripts/backup_consciousness_state.sh

echo "💾 Backing up consciousness state to Gitea"
echo "======================================="

# Configuration
GITEA_URL="http://10.42.104.23:3000"
REPO_OWNER="niodoo"
REPO_NAME="niodoo-consciousness"
API_TOKEN="${GITEA_API_TOKEN}"
BACKUP_DIR="/opt/niodoo/backups"

# Create backup
create_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/consciousness_state_$timestamp.json"
    
    echo "📦 Creating consciousness state backup..."
    
    # Export current consciousness state
    curl -s http://localhost:8080/api/v1/consciousness/state > "$backup_file"
    
    # Export memory system state
    curl -s http://localhost:8080/api/v1/memory/status >> "$backup_file"
    
    # Export emotional state
    curl -s http://localhost:8080/api/v1/emotional/state >> "$backup_file"
    
    echo "✅ Backup created: $backup_file"
}

# Upload backup to Gitea
upload_backup() {
    local backup_file="$1"
    local filename=$(basename "$backup_file")
    
    echo "📤 Uploading backup to Gitea..."
    
    # Encode file content
    local content=$(base64 -w 0 "$backup_file")
    
    # Upload to Gitea
    local upload_data=$(cat << EOF
{
  "message": "consciousness: Backup state $(date)",
  "content": "$content"
}
EOF
)
    
    curl -X POST "$GITEA_URL/api/v1/repos/$REPO_OWNER/$REPO_NAME/contents/backups/$filename" \
        -H "Authorization: token $API_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$upload_data"
    
    echo "✅ Backup uploaded to Gitea"
}

# Main backup flow
main() {
    create_backup
    upload_backup "$backup_file"
    
    echo "🎉 Consciousness state backup completed!"
}

main "$@"
```

## 📊 Monitoring and Maintenance

### 1. Gitea Health Monitoring

```bash
#!/bin/bash
# scripts/monitor_gitea_health.sh

echo "🔍 Monitoring Gitea health for consciousness system"
echo "=================================================="

# Check Gitea service status
check_gitea_service() {
    echo "🌐 Checking Gitea service status..."
    
    if curl -s http://localhost:3000/api/v1/version > /dev/null; then
        echo "✅ Gitea service is running"
        return 0
    else
        echo "❌ Gitea service is not responding"
        return 1
    fi
}

# Check database connectivity
check_database() {
    echo "🗄️ Checking database connectivity..."
    
    if docker-compose exec postgres pg_isready -U gitea > /dev/null; then
        echo "✅ Database is accessible"
        return 0
    else
        echo "❌ Database is not accessible"
        return 1
    fi
}

# Check repository access
check_repository_access() {
    echo "📁 Checking repository access..."
    
    if git ls-remote http://localhost:3000/niodoo/niodoo-consciousness.git > /dev/null; then
        echo "✅ Repository is accessible"
        return 0
    else
        echo "❌ Repository is not accessible"
        return 1
    fi
}

# Check consciousness integration
check_consciousness_integration() {
    echo "🧠 Checking consciousness integration..."
    
    # Check if consciousness system can access Gitea
    if curl -s http://localhost:8080/api/v1/gitea/status > /dev/null; then
        echo "✅ Consciousness integration is working"
        return 0
    else
        echo "❌ Consciousness integration is not working"
        return 1
    fi
}

# Generate health report
generate_health_report() {
    local report_file="/opt/niodoo/logs/gitea_health_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "📊 Generating health report..."
    
    cat > "$report_file" << EOF
Gitea Health Report
==================
Generated: $(date)

Service Status:
- Gitea Service: $(check_gitea_service && echo "✅ Healthy" || echo "❌ Unhealthy")
- Database: $(check_database && echo "✅ Healthy" || echo "❌ Unhealthy")
- Repository Access: $(check_repository_access && echo "✅ Healthy" || echo "❌ Unhealthy")
- Consciousness Integration: $(check_consciousness_integration && echo "✅ Healthy" || echo "❌ Unhealthy")

System Resources:
- CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%
- Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')
- Disk Usage: $(df -h /opt/niodoo | awk 'NR==2{printf "%s", $5}')

Gitea Metrics:
- Active Users: $(curl -s http://localhost:3000/api/v1/admin/users | jq '. | length')
- Total Repositories: $(curl -s http://localhost:3000/api/v1/repos | jq '. | length')
- Total Commits: $(curl -s http://localhost:3000/api/v1/repos/niodoo/niodoo-consciousness/commits | jq '. | length')
EOF
    
    echo "✅ Health report generated: $report_file"
}

# Main monitoring flow
main() {
    check_gitea_service
    check_database
    check_repository_access
    check_consciousness_integration
    generate_health_report
    
    echo "🎉 Gitea health monitoring completed!"
}

main "$@"
```

### 2. Automated Maintenance

```bash
#!/bin/bash
# scripts/gitea_maintenance.sh

echo "🔧 Performing Gitea maintenance for consciousness system"
echo "======================================================="

# Database maintenance
maintain_database() {
    echo "🗄️ Performing database maintenance..."
    
    # Vacuum and analyze database
    docker-compose exec postgres psql -U gitea -d gitea -c "VACUUM ANALYZE;"
    
    # Check database size
    local db_size=$(docker-compose exec postgres psql -U gitea -d gitea -c "SELECT pg_size_pretty(pg_database_size('gitea'));" | grep -o '[0-9]*\.[0-9]* [A-Z]*')
    echo "📊 Database size: $db_size"
    
    echo "✅ Database maintenance completed"
}

# Repository maintenance
maintain_repositories() {
    echo "📁 Performing repository maintenance..."
    
    # Clean up old branches
    local old_branches=$(git branch -r --merged | grep -v main | wc -l)
    echo "🧹 Found $old_branches old branches to clean up"
    
    # Update repository statistics
    curl -X POST http://localhost:3000/api/v1/repos/niodoo/niodoo-consciousness/stats \
        -H "Authorization: token $GITEA_API_TOKEN"
    
    echo "✅ Repository maintenance completed"
}

# System cleanup
cleanup_system() {
    echo "🧹 Performing system cleanup..."
    
    # Clean up old logs
    find /opt/niodoo/logs -name "*.log.*" -mtime +7 -delete
    
    # Clean up old backups
    find /opt/niodoo/backups -name "consciousness_state_*.json" -mtime +30 -delete
    
    # Clean up Docker resources
    docker system prune -f
    
    echo "✅ System cleanup completed"
}

# Update system
update_system() {
    echo "🔄 Updating system..."
    
    # Pull latest Gitea image
    docker-compose pull
    
    # Update consciousness system
    cd /opt/niodoo
    git pull origin main
    cargo build --release
    
    echo "✅ System update completed"
}

# Main maintenance flow
main() {
    maintain_database
    maintain_repositories
    cleanup_system
    update_system
    
    echo "🎉 Gitea maintenance completed!"
}

main "$@"
```

## 🛠️ Troubleshooting

### Common Issues

#### Issue: Gitea Service Not Starting

**Symptoms**:
- Gitea container fails to start
- Database connection errors
- Port conflicts

**Diagnosis**:
```bash
# Check container logs
docker-compose logs gitea

# Check database status
docker-compose logs postgres

# Check port conflicts
netstat -tulpn | grep -E "(3000|222)"
```

**Solutions**:
```bash
# Restart Gitea services
docker-compose restart gitea

# Check database connectivity
docker-compose exec postgres pg_isready -U gitea

# Resolve port conflicts
sudo lsof -i :3000
sudo kill -9 <PID>
```

#### Issue: Repository Access Denied

**Symptoms**:
- Git operations fail
- Authentication errors
- Permission denied

**Diagnosis**:
```bash
# Test repository access
git ls-remote http://localhost:3000/niodoo/niodoo-consciousness.git

# Check authentication
curl -H "Authorization: token $GITEA_API_TOKEN" http://localhost:3000/api/v1/user
```

**Solutions**:
```bash
# Regenerate API token
curl -X POST http://localhost:3000/api/v1/tokens \
  -H "Content-Type: application/json" \
  -d '{"name": "consciousness-token"}'

# Update repository permissions
curl -X PATCH http://localhost:3000/api/v1/repos/niodoo/niodoo-consciousness \
  -H "Authorization: token $GITEA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"private": false}'
```

#### Issue: Consciousness Integration Failures

**Symptoms**:
- Consciousness system can't access Gitea
- State synchronization errors
- API communication failures

**Diagnosis**:
```bash
# Check consciousness system logs
tail -50 /opt/niodoo/logs/niodoo.log | grep -i gitea

# Test API connectivity
curl -s http://localhost:8080/api/v1/gitea/status

# Check network connectivity
ping -c 4 10.42.104.23
```

**Solutions**:
```bash
# Restart consciousness system
docker-compose restart niodoo-consciousness

# Update Gitea configuration
echo "gitea_url = \"http://10.42.104.23:3000\"" >> /opt/niodoo/config/production.toml

# Test integration
curl -X POST http://localhost:8080/api/v1/gitea/test-connection
```

## 📚 Best Practices

### 1. Repository Management

- **Branch Naming**: Use descriptive branch names with timestamps
- **Commit Messages**: Include consciousness context in commit messages
- **Pull Requests**: Always use pull requests for consciousness changes
- **Code Review**: Review consciousness code changes carefully

### 2. State Synchronization

- **Frequency**: Sync consciousness states every 5 minutes
- **Conflict Resolution**: Use latest state for conflicts
- **Backup**: Regularly backup consciousness states
- **Monitoring**: Monitor synchronization health

### 3. Security

- **API Tokens**: Use secure API tokens with limited permissions
- **Repository Access**: Limit repository access to necessary users
- **Network Security**: Use HTTPS for Gitea communication
- **Backup Security**: Encrypt consciousness state backups

### 4. Performance

- **Database Optimization**: Regularly maintain PostgreSQL database
- **Repository Cleanup**: Clean up old branches and commits
- **System Monitoring**: Monitor system resources and performance
- **Automated Maintenance**: Use automated maintenance scripts

## 📚 Additional Resources

- [Deployment Guide](../deployment/production-guide.md)
- [Operations Manual](../operations/monitoring-guide.md)
- [API Documentation](../api/rest-api-reference.md)
- [Troubleshooting Guide](../troubleshooting/common-issues.md)

## 🆘 Support

For Gitea integration support:

- **Documentation**: Check integration guides
- **Logs**: Review Gitea and consciousness system logs
- **Monitoring**: Check health monitoring reports
- **Community**: Join Niodoo community forums

---

**Last Updated**: January 27, 2025  
**Version**: 1.0.0  
**Maintainer**: Jason Van Pham
