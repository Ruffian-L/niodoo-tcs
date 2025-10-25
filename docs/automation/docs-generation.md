# Automated Documentation Generation

## 🤖 Automated Documentation System for Niodoo

This document describes the automated documentation generation system for the Niodoo Consciousness Engine, including API documentation, architecture diagrams, and content generation.

## 📋 Table of Contents

- [Overview](#overview)
- [API Documentation Generation](#api-documentation-generation)
- [Architecture Diagram Generation](#architecture-diagram-generation)
- [Content Generation](#content-generation)
- [CI/CD Integration](#cicd-integration)
- [Configuration](#configuration)
- [Usage](#usage)

## 🌟 Overview

The automated documentation system provides:

- **API Documentation**: Auto-generated from Rust code comments
- **Architecture Diagrams**: Mermaid diagrams from code structure
- **Content Generation**: Automated content updates
- **CI/CD Integration**: Automated builds and deployments
- **Version Management**: Documentation versioning

## 🔧 API Documentation Generation

### Rust Documentation

```rust
//! # Niodoo Consciousness Engine
//! 
//! This crate provides the core consciousness engine for Niodoo AI systems.
//! 
//! ## Features
//! 
//! - Emergent consciousness processing
//! - Möbius topology mathematics
//! - Gaussian memory spheres
//! - Three-brain coordination
//! 
//! ## Example
//! 
//! ```rust
//! use niodoo_consciousness::PersonalNiodooConsciousness;
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let consciousness = PersonalNiodooConsciousness::new().await?;
//!     let response = consciousness.process_input("Hello, world!").await?;
//!     println!("Response: {}", response);
//!     Ok(())
//! }
//! ```

/// Processes user input through the consciousness engine
/// 
/// This function takes user input and processes it through all layers of the
/// consciousness engine, including emotional analysis, memory integration,
/// and brain coordination.
/// 
/// # Arguments
/// 
/// * `input` - The user input string to process
/// 
/// # Returns
/// 
/// * `Result<String>` - The processed response from the consciousness engine
/// 
/// # Errors
/// 
/// * `ConsciousnessError::BrainTimeout` - If brain processing times out
/// * `ConsciousnessError::MemoryError` - If memory operations fail
/// 
/// # Examples
/// 
/// ```rust
/// let consciousness = PersonalNiodooConsciousness::new().await?;
/// let response = consciousness.process_input("Hello, world!").await?;
/// println!("Response: {}", response);
/// ```
pub async fn process_input(&self, input: &str) -> Result<String> {
    // Implementation
}
```

### Documentation Generation Script

```bash
#!/bin/bash
# generate-api-docs.sh

set -e

echo "🔧 Generating API documentation..."

# Generate Rust documentation
cargo doc --no-deps --document-private-items

# Copy generated docs to docs/api/
cp -r target/doc/* docs/api/

# Generate API reference
cargo run --bin api-reference-generator -- --output docs/api/

# Generate examples documentation
cargo run --bin examples-generator -- --output docs/examples/

echo "✅ API documentation generated successfully"
```

## 📊 Architecture Diagram Generation

### Mermaid Diagram Generator

```rust
// src/tools/architecture_generator.rs

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

pub struct ArchitectureGenerator {
    components: Vec<Component>,
    relationships: Vec<Relationship>,
}

pub struct Component {
    pub name: String,
    pub component_type: ComponentType,
    pub description: String,
    pub dependencies: Vec<String>,
}

pub enum ComponentType {
    ConsciousnessEngine,
    MemoryManager,
    BrainCoordinator,
    MobiusEngine,
    Phase6Integration,
}

pub struct Relationship {
    pub from: String,
    pub to: String,
    pub relationship_type: RelationshipType,
}

pub enum RelationshipType {
    Uses,
    Contains,
    CommunicatesWith,
    DependsOn,
}

impl ArchitectureGenerator {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            relationships: Vec::new(),
        }
    }
    
    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }
    
    pub fn add_relationship(&mut self, relationship: Relationship) {
        self.relationships.push(relationship);
    }
    
    pub fn generate_mermaid_diagram(&self) -> String {
        let mut diagram = String::new();
        
        diagram.push_str("graph TB\n");
        
        // Add components
        for component in &self.components {
            let node_id = self.sanitize_id(&component.name);
            let label = format!("{}[{}]", node_id, component.name);
            diagram.push_str(&format!("    {}\n", label));
        }
        
        // Add relationships
        for relationship in &self.relationships {
            let from_id = self.sanitize_id(&relationship.from);
            let to_id = self.sanitize_id(&relationship.to);
            let arrow = match relationship.relationship_type {
                RelationshipType::Uses => "-->",
                RelationshipType::Contains => "-->",
                RelationshipType::CommunicatesWith => "<-->",
                RelationshipType::DependsOn => "-->",
            };
            diagram.push_str(&format!("    {} {} {}\n", from_id, arrow, to_id));
        }
        
        diagram
    }
    
    fn sanitize_id(&self, name: &str) -> String {
        name.replace(" ", "_").replace("-", "_").to_lowercase()
    }
    
    pub fn generate_diagram_file(&self, output_path: &str) -> Result<()> {
        let diagram = self.generate_mermaid_diagram();
        let mut file = File::create(output_path)?;
        file.write_all(diagram.as_bytes())?;
        Ok(())
    }
}
```

### Diagram Generation Script

```bash
#!/bin/bash
# generate-architecture-diagrams.sh

set -e

echo "📊 Generating architecture diagrams..."

# Generate system overview diagram
cargo run --bin architecture-generator -- --output docs/architecture/system-overview.mmd

# Generate consciousness engine diagram
cargo run --bin architecture-generator -- --component consciousness-engine --output docs/architecture/consciousness-engine.mmd

# Generate brain coordination diagram
cargo run --bin architecture-generator -- --component brain-coordination --output docs/architecture/brain-coordination.mmd

# Generate memory management diagram
cargo run --bin architecture-generator -- --component memory-management --output docs/architecture/memory-management.mmd

# Generate Phase 6 integration diagram
cargo run --bin architecture-generator -- --component phase6-integration --output docs/architecture/phase6-integration.mmd

echo "✅ Architecture diagrams generated successfully"
```

## 📝 Content Generation

### Content Generator

```rust
// src/tools/content_generator.rs

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

pub struct ContentGenerator {
    templates: HashMap<String, String>,
    data_sources: Vec<Box<dyn DataSource>>,
}

pub trait DataSource {
    fn get_data(&self) -> Result<HashMap<String, String>>;
}

pub struct CodeDataSource {
    pub source_path: String,
}

impl DataSource for CodeDataSource {
    fn get_data(&self) -> Result<HashMap<String, String>> {
        // Extract data from source code
        let mut data = HashMap::new();
        
        // Parse Rust source files
        let source_files = self.find_rust_files()?;
        for file in source_files {
            let content = std::fs::read_to_string(&file)?;
            let parsed_data = self.parse_rust_file(&content)?;
            data.extend(parsed_data);
        }
        
        Ok(data)
    }
}

impl ContentGenerator {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            data_sources: Vec::new(),
        }
    }
    
    pub fn add_template(&mut self, name: String, template: String) {
        self.templates.insert(name, template);
    }
    
    pub fn add_data_source(&mut self, source: Box<dyn DataSource>) {
        self.data_sources.push(source);
    }
    
    pub fn generate_content(&self, template_name: &str, output_path: &str) -> Result<()> {
        let template = self.templates.get(template_name)
            .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_name))?;
        
        // Collect data from all sources
        let mut all_data = HashMap::new();
        for source in &self.data_sources {
            let data = source.get_data()?;
            all_data.extend(data);
        }
        
        // Render template with data
        let rendered = self.render_template(template, &all_data)?;
        
        // Write to output file
        let mut file = File::create(output_path)?;
        file.write_all(rendered.as_bytes())?;
        
        Ok(())
    }
    
    fn render_template(&self, template: &str, data: &HashMap<String, String>) -> Result<String> {
        let mut rendered = template.to_string();
        
        // Simple template rendering (replace {{key}} with values)
        for (key, value) in data {
            let placeholder = format!("{{{{{}}}}}", key);
            rendered = rendered.replace(&placeholder, value);
        }
        
        Ok(rendered)
    }
}
```

### Content Generation Script

```bash
#!/bin/bash
# generate-content.sh

set -e

echo "📝 Generating content..."

# Generate API reference
cargo run --bin content-generator -- --template api-reference --output docs/api/

# Generate examples
cargo run --bin content-generator -- --template examples --output docs/examples/

# Generate troubleshooting guides
cargo run --bin content-generator -- --template troubleshooting --output docs/troubleshooting/

# Generate user guides
cargo run --bin content-generator -- --template user-guides --output docs/user-guides/

echo "✅ Content generated successfully"
```

## 🔄 CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/docs.yml

name: Documentation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: rustfmt, clippy, rust-src
    
    - name: Cache cargo registry
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install build-essential pkg-config libssl-dev
    
    - name: Generate API documentation
      run: |
        cargo doc --no-deps --document-private-items
        cp -r target/doc/* docs/api/
    
    - name: Generate architecture diagrams
      run: |
        cargo run --bin architecture-generator -- --output docs/architecture/
    
    - name: Generate content
      run: |
        cargo run --bin content-generator -- --output docs/
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs
```

### GitLab CI Pipeline

```yaml
# .gitlab-ci.yml

stages:
  - build
  - test
  - docs
  - deploy

variables:
  CARGO_HOME: $CI_PROJECT_DIR/cargo
  CARGO_TARGET_DIR: $CI_PROJECT_DIR/target

cache:
  paths:
    - cargo/
    - target/

build:
  stage: build
  image: rust:1.70
  script:
    - cargo build --release
  artifacts:
    paths:
      - target/release/
    expire_in: 1 hour

test:
  stage: test
  image: rust:1.70
  script:
    - cargo test
    - cargo clippy -- -D warnings
    - cargo fmt --check

docs:
  stage: docs
  image: rust:1.70
  script:
    - cargo doc --no-deps --document-private-items
    - cargo run --bin architecture-generator -- --output docs/architecture/
    - cargo run --bin content-generator -- --output docs/
  artifacts:
    paths:
      - docs/
    expire_in: 1 week

deploy:
  stage: deploy
  image: alpine:latest
  script:
    - apk add --no-cache rsync
    - rsync -av docs/ $DEPLOY_HOST:/var/www/niodoo-docs/
  only:
    - main
```

## ⚙️ Configuration

### Documentation Configuration

```yaml
# docs-config.yaml

documentation:
  api:
    enabled: true
    output_dir: "docs/api"
    include_private: true
    theme: "ayu"
  
  architecture:
    enabled: true
    output_dir: "docs/architecture"
    formats: ["mermaid", "svg", "png"]
  
  content:
    enabled: true
    output_dir: "docs"
    templates_dir: "templates"
    data_sources:
      - type: "code"
        path: "src/"
      - type: "config"
        path: "config/"
  
  deployment:
    enabled: true
    target: "github-pages"
    branch: "gh-pages"
    domain: "niodoo.github.io"
```

### Template Configuration

```yaml
# templates/api-reference.yaml

template: |
  # API Reference
  
  ## {{component_name}}
  
  {{component_description}}
  
  ### Methods
  
  {{#methods}}
  #### {{method_name}}
  
  {{method_description}}
  
  **Parameters:**
  {{#parameters}}
  - `{{name}}`: {{description}}
  {{/parameters}}
  
  **Returns:**
  {{returns}}
  
  **Example:**
  ```rust
  {{example}}
  ```
  
  {{/methods}}

data_mapping:
  component_name: "consciousness_engine"
  component_description: "description"
  methods: "methods"
```

## 🚀 Usage

### Manual Generation

```bash
# Generate all documentation
./scripts/generate-all-docs.sh

# Generate specific documentation
./scripts/generate-api-docs.sh
./scripts/generate-architecture-diagrams.sh
./scripts/generate-content.sh
```

### Automated Generation

```bash
# Set up automated generation
cargo install --path tools/docs-generator

# Run automated generation
docs-generator --config docs-config.yaml --output docs/
```

### Development Workflow

```bash
# During development
cargo run --bin docs-generator -- --watch --output docs/

# Before commit
./scripts/generate-all-docs.sh
git add docs/
git commit -m "Update documentation"
```

## 📚 Additional Resources

### Tools
- [Rust Documentation](https://doc.rust-lang.org/book/ch14-02-publishing-to-crates-io.html) - Rust documentation system
- [Mermaid](https://mermaid-js.github.io/mermaid/) - Diagram generation
- [GitHub Pages](https://pages.github.com/) - Documentation hosting
- [GitLab Pages](https://docs.gitlab.com/ee/user/project/pages/) - Documentation hosting

### Templates
- [API Reference Template](../../templates/api-reference.md) - API documentation template
- [Architecture Template](../../templates/architecture.md) - Architecture documentation template
- [User Guide Template](../../templates/user-guide.md) - User guide template

---

*This document describes the automated documentation generation system for the Niodoo Consciousness Engine. For implementation details, refer to the source code in `src/tools/`.*
