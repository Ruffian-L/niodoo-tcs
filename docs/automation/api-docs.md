# API Documentation Automation

## 🔧 Automated API Documentation Generation

This document describes the automated API documentation generation system for the Niodoo Consciousness Engine, including Rust documentation, OpenAPI specifications, and interactive API docs.

## 📋 Table of Contents

- [Overview](#overview)
- [Rust Documentation](#rust-documentation)
- [OpenAPI Generation](#openapi-generation)
- [Interactive Documentation](#interactive-documentation)
- [CI/CD Integration](#cicd-integration)
- [Configuration](#configuration)
- [Usage](#usage)

## 🌟 Overview

The API documentation automation system provides:

- **Rust Documentation**: Auto-generated from code comments
- **OpenAPI Specifications**: REST API documentation
- **Interactive Documentation**: Swagger UI integration
- **Code Examples**: Automated example generation
- **Version Management**: API versioning and documentation

## 📚 Rust Documentation

### Documentation Comments

```rust
//! # Niodoo Consciousness Engine API
//! 
//! This crate provides the core API for the Niodoo Consciousness Engine.
//! 
//! ## Features
//! 
//! - Emergent consciousness processing
//! - Möbius topology mathematics
//! - Gaussian memory spheres
//! - Three-brain coordination
//! 
//! ## Quick Start
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

### Documentation Generation

```bash
#!/bin/bash
# generate-rust-docs.sh

set -e

echo "📚 Generating Rust documentation..."

# Generate documentation
cargo doc --no-deps --document-private-items --open

# Copy to docs directory
cp -r target/doc/* docs/api/rust/

# Generate index page
cat > docs/api/rust/index.md << EOF
# Rust API Documentation

This is the auto-generated Rust API documentation for the Niodoo Consciousness Engine.

## Modules

- [consciousness_engine](consciousness_engine/index.html) - Core consciousness engine
- [memory_management](memory_management/index.html) - Memory management system
- [brain_coordination](brain_coordination/index.html) - Brain coordination system
- [mobius_topology](mobius_topology/index.html) - Möbius topology mathematics

## Examples

See the [examples](../examples/) directory for usage examples.

EOF

echo "✅ Rust documentation generated successfully"
```

## 🌐 OpenAPI Generation

### OpenAPI Specification

```yaml
# openapi.yaml

openapi: 3.0.3
info:
  title: Niodoo Consciousness Engine API
  description: API for the Niodoo Consciousness Engine
  version: 1.0.0
  contact:
    name: Niodoo Team
    url: https://github.com/niodoo/niodoo-feeling
    email: team@niodoo.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: http://localhost:8080/api/v1
    description: Development server
  - url: https://api.niodoo.com/v1
    description: Production server

paths:
  /consciousness/process:
    post:
      summary: Process input through consciousness engine
      description: Processes user input through the consciousness engine
      operationId: processInput
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ProcessInputRequest'
      responses:
        '200':
          description: Successful processing
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ProcessInputResponse'
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '500':
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /consciousness/emotional-state:
    get:
      summary: Get current emotional state
      description: Retrieves the current emotional state
      operationId: getEmotionalState
      responses:
        '200':
          description: Current emotional state
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EmotionalState'
        '500':
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

    put:
      summary: Update emotional state
      description: Updates the current emotional state
      operationId: updateEmotionalState
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EmotionalState'
      responses:
        '200':
          description: Emotional state updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EmotionalState'
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /memory/spheres:
    get:
      summary: List memory spheres
      description: Retrieves a list of memory spheres
      operationId: listMemorySpheres
      parameters:
        - name: limit
          in: query
          description: Maximum number of spheres to return
          required: false
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 10
        - name: offset
          in: query
          description: Number of spheres to skip
          required: false
          schema:
            type: integer
            minimum: 0
            default: 0
      responses:
        '200':
          description: List of memory spheres
          content:
            application/json:
              schema:
                type: object
                properties:
                  spheres:
                    type: array
                    items:
                      $ref: '#/components/schemas/MemorySphere'
                  total:
                    type: integer
                    description: Total number of spheres
                  limit:
                    type: integer
                    description: Maximum number of spheres returned
                  offset:
                    type: integer
                    description: Number of spheres skipped

    post:
      summary: Create memory sphere
      description: Creates a new memory sphere
      operationId: createMemorySphere
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateMemorySphereRequest'
      responses:
        '201':
          description: Memory sphere created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MemorySphere'
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    ProcessInputRequest:
      type: object
      required:
        - input
      properties:
        input:
          type: string
          description: The input text to process
          example: "Hello, how are you feeling today?"
        context:
          type: string
          description: Additional context information
          example: "User is feeling anxious about work deadline"
        emotional_context:
          type: array
          items:
            type: number
            format: float
          minItems: 4
          maxItems: 4
          description: Emotional context [joy, sadness, anger, fear]
          example: [0.8, 0.1, 0.0, 0.2]

    ProcessInputResponse:
      type: object
      required:
        - response
        - processing_time
      properties:
        response:
          type: string
          description: The processed response
          example: "I understand you're feeling anxious. Let me help you work through this."
        processing_time:
          type: integer
          format: int64
          description: Processing time in milliseconds
          example: 1500
        emotional_state:
          $ref: '#/components/schemas/EmotionalState'
        brain_activity:
          type: object
          description: Brain activity levels
          properties:
            motor:
              type: number
              format: float
              description: Motor brain activity level
            lcars:
              type: number
              format: float
              description: LCARS brain activity level
            efficiency:
              type: number
              format: float
              description: Efficiency brain activity level

    EmotionalState:
      type: object
      required:
        - joy
        - sadness
        - anger
        - fear
      properties:
        joy:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          description: Joy level
          example: 0.8
        sadness:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          description: Sadness level
          example: 0.1
        anger:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          description: Anger level
          example: 0.0
        fear:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          description: Fear level
          example: 0.2

    MemorySphere:
      type: object
      required:
        - id
        - content
        - position
        - emotional_valence
      properties:
        id:
          type: string
          description: Unique identifier
          example: "sphere_12345"
        content:
          type: string
          description: Memory content
          example: "I learned about quantum computing today"
        position:
          type: array
          items:
            type: number
            format: float
          minItems: 3
          maxItems: 3
          description: 3D position [x, y, z]
          example: [1.5, 2.3, 0.8]
        emotional_valence:
          type: number
          format: float
          minimum: -1.0
          maximum: 1.0
          description: Emotional valence
          example: 0.7
        creation_time:
          type: string
          format: date-time
          description: Creation timestamp
          example: "2023-01-01T12:00:00Z"
        access_count:
          type: integer
          description: Number of times accessed
          example: 42
        last_accessed:
          type: string
          format: date-time
          description: Last access timestamp
          example: "2023-01-01T15:30:00Z"

    CreateMemorySphereRequest:
      type: object
      required:
        - content
        - emotional_valence
      properties:
        content:
          type: string
          description: Memory content
          example: "I learned about quantum computing today"
        emotional_valence:
          type: number
          format: float
          minimum: -1.0
          maximum: 1.0
          description: Emotional valence
          example: 0.7
        context:
          type: string
          description: Additional context
          example: "Educational context"

    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
          description: Error type
          example: "ConsciousnessError"
        message:
          type: string
          description: Error message
          example: "Brain processing timeout"
        code:
          type: integer
          description: Error code
          example: 1001
        details:
          type: object
          description: Additional error details

  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - ApiKeyAuth: []
  - BearerAuth: []
```

### OpenAPI Generation Script

```bash
#!/bin/bash
# generate-openapi.sh

set -e

echo "🌐 Generating OpenAPI specification..."

# Generate OpenAPI spec from Rust code
cargo run --bin openapi-generator -- --output docs/api/openapi.yaml

# Validate OpenAPI spec
npx @apidevtools/swagger-parser validate docs/api/openapi.yaml

# Generate OpenAPI client
npx @openapitools/openapi-generator-cli generate \
  -i docs/api/openapi.yaml \
  -g rust \
  -o clients/rust-client

# Generate OpenAPI server stub
npx @openapitools/openapi-generator-cli generate \
  -i docs/api/openapi.yaml \
  -g rust-server \
  -o servers/rust-server

echo "✅ OpenAPI specification generated successfully"
```

## 🎨 Interactive Documentation

### Swagger UI Integration

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Niodoo Consciousness Engine API</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }
        body {
            margin:0;
            background: #fafafa;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: './openapi.yaml',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                onComplete: function() {
                    console.log('Swagger UI loaded');
                }
            });
        };
    </script>
</body>
</html>
```

### Interactive Documentation Generator

```rust
// src/tools/interactive_docs_generator.rs

use std::fs::File;
use std::io::Write;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiEndpoint {
    pub path: String,
    pub method: String,
    pub summary: String,
    pub description: String,
    pub parameters: Vec<ApiParameter>,
    pub responses: Vec<ApiResponse>,
    pub examples: Vec<ApiExample>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiParameter {
    pub name: String,
    pub parameter_type: String,
    pub required: bool,
    pub description: String,
    pub example: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse {
    pub status_code: u16,
    pub description: String,
    pub schema: Option<String>,
    pub example: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiExample {
    pub name: String,
    pub description: String,
    pub request: String,
    pub response: String,
}

pub struct InteractiveDocsGenerator {
    pub endpoints: Vec<ApiEndpoint>,
    pub base_url: String,
}

impl InteractiveDocsGenerator {
    pub fn new(base_url: String) -> Self {
        Self {
            endpoints: Vec::new(),
            base_url,
        }
    }
    
    pub fn add_endpoint(&mut self, endpoint: ApiEndpoint) {
        self.endpoints.push(endpoint);
    }
    
    pub fn generate_html(&self) -> String {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        html.push_str("    <title>Niodoo Consciousness Engine API</title>\n");
        html.push_str("    <link rel=\"stylesheet\" type=\"text/css\" href=\"https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css\" />\n");
        html.push_str("    <style>\n");
        html.push_str("        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }\n");
        html.push_str("        *, *:before, *:after { box-sizing: inherit; }\n");
        html.push_str("        body { margin:0; background: #fafafa; }\n");
        html.push_str("    </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");
        html.push_str("    <div id=\"swagger-ui\"></div>\n");
        html.push_str("    <script src=\"https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js\"></script>\n");
        html.push_str("    <script src=\"https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js\"></script>\n");
        html.push_str("    <script>\n");
        html.push_str("        window.onload = function() {\n");
        html.push_str("            const ui = SwaggerUIBundle({\n");
        html.push_str("                url: './openapi.yaml',\n");
        html.push_str("                dom_id: '#swagger-ui',\n");
        html.push_str("                deepLinking: true,\n");
        html.push_str("                presets: [\n");
        html.push_str("                    SwaggerUIBundle.presets.apis,\n");
        html.push_str("                    SwaggerUIStandalonePreset\n");
        html.push_str("                ],\n");
        html.push_str("                plugins: [\n");
        html.push_str("                    SwaggerUIBundle.plugins.DownloadUrl\n");
        html.push_str("                ],\n");
        html.push_str("                layout: \"StandaloneLayout\",\n");
        html.push_str("                validatorUrl: null\n");
        html.push_str("            });\n");
        html.push_str("        };\n");
        html.push_str("    </script>\n");
        html.push_str("</body>\n");
        html.push_str("</html>\n");
        
        html
    }
    
    pub fn generate_examples(&self) -> String {
        let mut examples = String::new();
        
        examples.push_str("# API Examples\n\n");
        
        for endpoint in &self.endpoints {
            examples.push_str(&format!("## {} {}\n\n", endpoint.method, endpoint.path));
            examples.push_str(&format!("**Summary:** {}\n\n", endpoint.summary));
            examples.push_str(&format!("**Description:** {}\n\n", endpoint.description));
            
            if !endpoint.parameters.is_empty() {
                examples.push_str("### Parameters\n\n");
                for param in &endpoint.parameters {
                    examples.push_str(&format!("- **{}** (`{}`): {}\n", 
                        param.name, param.parameter_type, param.description));
                    if let Some(example) = &param.example {
                        examples.push_str(&format!("  - Example: `{}`\n", example));
                    }
                }
                examples.push_str("\n");
            }
            
            if !endpoint.examples.is_empty() {
                examples.push_str("### Examples\n\n");
                for example in &endpoint.examples {
                    examples.push_str(&format!("#### {}\n\n", example.name));
                    examples.push_str(&format!("**Description:** {}\n\n", example.description));
                    examples.push_str("**Request:**\n");
                    examples.push_str("```bash\n");
                    examples.push_str(&example.request);
                    examples.push_str("\n```\n\n");
                    examples.push_str("**Response:**\n");
                    examples.push_str("```json\n");
                    examples.push_str(&example.response);
                    examples.push_str("\n```\n\n");
                }
            }
            
            examples.push_str("---\n\n");
        }
        
        examples
    }
    
    pub fn generate_docs(&self, output_dir: &str) -> Result<()> {
        // Create output directory
        std::fs::create_dir_all(output_dir)?;
        
        // Generate HTML
        let html = self.generate_html();
        let mut file = File::create(format!("{}/index.html", output_dir))?;
        file.write_all(html.as_bytes())?;
        
        // Generate examples
        let examples = self.generate_examples();
        let mut file = File::create(format!("{}/examples.md", output_dir))?;
        file.write_all(examples.as_bytes())?;
        
        Ok(())
    }
}
```

## 🔄 CI/CD Integration

### GitHub Actions for API Docs

```yaml
# .github/workflows/api-docs.yml

name: API Documentation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  generate-api-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        components: rustfmt, clippy, rust-src
    
    - name: Install Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install build-essential pkg-config libssl-dev
        npm install -g @openapitools/openapi-generator-cli
        npm install -g @apidevtools/swagger-parser
    
    - name: Generate Rust documentation
      run: |
        cargo doc --no-deps --document-private-items
        cp -r target/doc/* docs/api/rust/
    
    - name: Generate OpenAPI specification
      run: |
        cargo run --bin openapi-generator -- --output docs/api/openapi.yaml
        npx @apidevtools/swagger-parser validate docs/api/openapi.yaml
    
    - name: Generate interactive documentation
      run: |
        cargo run --bin interactive-docs-generator -- --output docs/api/interactive/
    
    - name: Generate API clients
      run: |
        npx @openapitools/openapi-generator-cli generate \
          -i docs/api/openapi.yaml \
          -g rust \
          -o clients/rust-client
        npx @openapitools/openapi-generator-cli generate \
          -i docs/api/openapi.yaml \
          -g python \
          -o clients/python-client
        npx @openapitools/openapi-generator-cli generate \
          -i docs/api/openapi.yaml \
          -g javascript \
          -o clients/javascript-client
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs
```

## ⚙️ Configuration

### API Documentation Configuration

```yaml
# api-docs-config.yaml

api_documentation:
  rust:
    enabled: true
    output_dir: "docs/api/rust"
    include_private: true
    theme: "ayu"
    generate_examples: true
  
  openapi:
    enabled: true
    output_file: "docs/api/openapi.yaml"
    version: "3.0.3"
    base_url: "http://localhost:8080/api/v1"
    generate_clients: true
    client_languages: ["rust", "python", "javascript"]
  
  interactive:
    enabled: true
    output_dir: "docs/api/interactive"
    swagger_ui_version: "4.15.5"
    enable_try_it_out: true
    enable_authentication: true
  
  examples:
    enabled: true
    output_dir: "docs/api/examples"
    languages: ["rust", "python", "javascript", "bash"]
    include_curl: true
    include_postman: true
```

## 🚀 Usage

### Manual Generation

```bash
# Generate all API documentation
./scripts/generate-api-docs.sh

# Generate specific documentation
./scripts/generate-rust-docs.sh
./scripts/generate-openapi.sh
./scripts/generate-interactive-docs.sh
```

### Automated Generation

```bash
# Set up automated generation
cargo install --path tools/api-docs-generator

# Run automated generation
api-docs-generator --config api-docs-config.yaml --output docs/api/
```

### Development Workflow

```bash
# During development
cargo run --bin api-docs-generator -- --watch --output docs/api/

# Before commit
./scripts/generate-api-docs.sh
git add docs/api/
git commit -m "Update API documentation"
```

## 📚 Additional Resources

### Tools
- [Rust Documentation](https://doc.rust-lang.org/book/ch14-02-publishing-to-crates-io.html) - Rust documentation system
- [OpenAPI Generator](https://openapi-generator.tech/) - OpenAPI client/server generation
- [Swagger UI](https://swagger.io/tools/swagger-ui/) - Interactive API documentation
- [Postman](https://www.postman.com/) - API testing and documentation

### Templates
- [OpenAPI Template](../../templates/openapi.yaml) - OpenAPI specification template
- [Swagger UI Template](../../templates/swagger-ui.html) - Swagger UI template
- [API Examples Template](../../templates/api-examples.md) - API examples template

---

*This document describes the automated API documentation generation system for the Niodoo Consciousness Engine. For implementation details, refer to the source code in `src/tools/`.*
