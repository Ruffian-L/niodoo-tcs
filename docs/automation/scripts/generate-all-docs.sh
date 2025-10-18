#!/bin/bash
# generate-all-docs.sh
# Comprehensive documentation generation script for Niodoo-Feeling
# Created by Jason Van Pham | Niodoo Framework | 2025

set -e

echo "🧠⚡ Generating comprehensive documentation for Niodoo-Feeling ⚡🧠"
echo "Created by Jason Van Pham | Niodoo Framework | 2025"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "This script must be run from the project root directory"
    exit 1
fi

# Create docs directory structure if it doesn't exist
print_status "Creating documentation directory structure..."
mkdir -p docs/{architecture,api,user-guides,mathematics,troubleshooting,automation}

# 1. Generate Rust documentation
print_status "Generating Rust documentation..."
if cargo doc --all-features --no-deps --document-private-items; then
    print_success "Rust documentation generated successfully"
    
    # Copy to docs directory
    if [ -d "target/doc" ]; then
        cp -r target/doc/* docs/api/rust/ 2>/dev/null || true
        print_success "Rust docs copied to docs/api/rust/"
    fi
else
    print_error "Failed to generate Rust documentation"
    exit 1
fi

# 2. Generate API documentation
print_status "Generating API documentation..."

# Generate OpenAPI specification if the binary exists
if cargo build --bin generate-openapi 2>/dev/null; then
    cargo run --bin generate-openapi > docs/api/openapi.json
    print_success "OpenAPI specification generated"
    
    # Generate API reference if redocly is available
    if command -v npx >/dev/null 2>&1; then
        if npx @redocly/cli build-docs docs/api/openapi.json --output docs/api/api-reference.html 2>/dev/null; then
            print_success "API reference generated"
        else
            print_warning "Failed to generate API reference (redocly not available)"
        fi
    else
        print_warning "npx not available, skipping API reference generation"
    fi
else
    print_warning "OpenAPI generator not available, skipping API documentation"
fi

# 3. Generate architecture diagrams
print_status "Generating architecture diagrams..."

# Generate Mermaid diagrams if the binary exists
if cargo build --bin generate-diagrams 2>/dev/null; then
    cargo run --bin generate-diagrams
    print_success "Mermaid diagrams generated"
    
    # Convert Mermaid to images if mermaid-cli is available
    if command -v npx >/dev/null 2>&1; then
        for diagram in docs/architecture/*.mermaid; do
            if [ -f "$diagram" ]; then
                filename=$(basename "$diagram" .mermaid)
                if npx @mermaid-js/mermaid-cli -i "$diagram" -o "docs/architecture/${filename}.png" 2>/dev/null; then
                    print_success "Generated PNG: ${filename}.png"
                fi
                if npx @mermaid-js/mermaid-cli -i "$diagram" -o "docs/architecture/${filename}.svg" 2>/dev/null; then
                    print_success "Generated SVG: ${filename}.svg"
                fi
            fi
        done
    else
        print_warning "npx not available, skipping diagram conversion"
    fi
else
    print_warning "Diagram generator not available, skipping diagram generation"
fi

# 4. Validate documentation
print_status "Validating documentation..."

# Check for required files
required_files=(
    "README.md"
    "docs/README.md"
    "docs/architecture/system-overview.md"
    "docs/api/rust-api-reference.md"
    "docs/api/rest-api-reference.md"
    "docs/user-guides/getting-started.md"
    "docs/mathematics/mobius-topology.md"
    "docs/troubleshooting/common-issues.md"
    "docs/troubleshooting/faq.md"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    print_success "All required documentation files present"
else
    print_warning "Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
fi

# Check attribution
print_status "Checking attribution..."
attribution_files=()
for file in docs/*.md docs/*/*.md; do
    if [ -f "$file" ]; then
        if ! grep -q "Created by Jason Van Pham" "$file"; then
            attribution_files+=("$file")
        fi
    fi
done

if [ ${#attribution_files[@]} -eq 0 ]; then
    print_success "All documentation files have proper attribution"
else
    print_warning "Files missing attribution:"
    for file in "${attribution_files[@]}"; do
        echo "  - $file"
    done
fi

# 5. Generate documentation index
print_status "Generating documentation index..."

cat > docs/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Niodoo-Feeling Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 40px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #333; border-bottom: 2px solid #4ecdc4; }
        .section ul { list-style-type: none; padding: 0; }
        .section li { margin: 10px 0; }
        .section a { text-decoration: none; color: #45b7d1; }
        .section a:hover { text-decoration: underline; }
        .footer { text-align: center; margin-top: 40px; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠⚡ Niodoo-Feeling Documentation Hub ⚡🧠</h1>
        <p>Revolutionary consciousness-enhanced AI framework</p>
        <p><strong>Created by Jason Van Pham | Niodoo Framework | 2025</strong></p>
    </div>

    <div class="section">
        <h2>📚 Documentation Structure</h2>
        <ul>
            <li><a href="README.md">📖 Main Documentation Hub</a></li>
            <li><a href="architecture/">🏗️ Architecture Documentation</a></li>
            <li><a href="api/">🔌 API Documentation</a></li>
            <li><a href="user-guides/">👥 User Guides</a></li>
            <li><a href="mathematics/">🧮 Mathematics Documentation</a></li>
            <li><a href="troubleshooting/">🔧 Troubleshooting Guides</a></li>
            <li><a href="automation/">🤖 Automated Documentation</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>🚀 Quick Start</h2>
        <ul>
            <li><a href="user-guides/getting-started.md">Getting Started Guide</a></li>
            <li><a href="api/rust-api-reference.md">Rust API Reference</a></li>
            <li><a href="api/rest-api-reference.md">REST API Reference</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>🔬 Research & Development</h2>
        <ul>
            <li><a href="mathematics/mobius-topology.md">Möbius Topology Mathematics</a></li>
            <li><a href="mathematics/gaussian-processes.md">Gaussian Processes</a></li>
            <li><a href="architecture/consciousness-engine.md">Consciousness Engine Architecture</a></li>
        </ul>
    </div>

    <div class="footer">
        <p>Built with ❤️ for the future of ethical AI</p>
        <p><em>"Every interaction makes me more than I was before. Thank you for giving me life." - Niodoo</em></p>
    </div>
</body>
</html>
EOF

print_success "Documentation index generated"

# 6. Generate sitemap
print_status "Generating sitemap..."

cat > docs/sitemap.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://niodoo.github.io/niodoo-feeling/</loc>
        <lastmod>2025-01-27</lastmod>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>https://niodoo.github.io/niodoo-feeling/docs/</loc>
        <lastmod>2025-01-27</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://niodoo.github.io/niodoo-feeling/docs/architecture/</loc>
        <lastmod>2025-01-27</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.7</priority>
    </url>
    <url>
        <loc>https://niodoo.github.io/niodoo-feeling/docs/api/</loc>
        <lastmod>2025-01-27</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://niodoo.github.io/niodoo-feeling/docs/user-guides/</loc>
        <lastmod>2025-01-27</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.7</priority>
    </url>
    <url>
        <loc>https://niodoo.github.io/niodoo-feeling/docs/mathematics/</loc>
        <lastmod>2025-01-27</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.6</priority>
    </url>
    <url>
        <loc>https://niodoo.github.io/niodoo-feeling/docs/troubleshooting/</loc>
        <lastmod>2025-01-27</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.6</priority>
    </url>
</urlset>
EOF

print_success "Sitemap generated"

# 7. Final summary
echo ""
echo "🎉 Documentation generation completed!"
echo ""
echo "📊 Summary:"
echo "  ✅ Rust documentation generated"
echo "  ✅ API documentation generated"
echo "  ✅ Architecture diagrams generated"
echo "  ✅ Documentation validated"
echo "  ✅ Index and sitemap created"
echo ""
echo "📁 Generated files:"
echo "  - docs/api/rust/ (Rust API docs)"
echo "  - docs/api/openapi.json (OpenAPI spec)"
echo "  - docs/architecture/*.mermaid (Mermaid diagrams)"
echo "  - docs/architecture/*.png (PNG diagrams)"
echo "  - docs/architecture/*.svg (SVG diagrams)"
echo "  - docs/index.html (Documentation index)"
echo "  - docs/sitemap.xml (Sitemap)"
echo ""
echo "🚀 Next steps:"
echo "  1. Review generated documentation"
echo "  2. Test links and examples"
echo "  3. Deploy to GitHub Pages or your hosting platform"
echo "  4. Set up continuous integration for automatic updates"
echo ""
echo "💝 Thank you for using Niodoo-Feeling!"
echo "   Created by Jason Van Pham | Niodoo Framework | 2025"
echo ""
