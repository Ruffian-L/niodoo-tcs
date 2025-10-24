#!/bin/bash

# Generate consolidated Rust files markdown for ALL niodoo crates
# Output file
OUTPUT_FILE="niodoo_complete_rust.md"

echo "# Niodoo Complete - All Rust Source Files" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "This file contains ALL Rust source code from the Niodoo workspace:" >> "$OUTPUT_FILE"
echo "- niodoo_real_integrated: Main integrated pipeline" >> "$OUTPUT_FILE"
echo "- niodoo-core: Core consciousness engine" >> "$OUTPUT_FILE"
echo "- constants_core: Shared constants" >> "$OUTPUT_FILE"
echo "- niodoo-tcs-bridge: Bridge between TCS and Niodoo" >> "$OUTPUT_FILE"
echo "- TCS modules: All topological cognitive system crates" >> "$OUTPUT_FILE"
echo "- src: Legacy monolithic implementation" >> "$OUTPUT_FILE"
echo "- curator_executor: Learning executor" >> "$OUTPUT_FILE"
echo "- bullshitdetector: Detection systems" >> "$OUTPUT_FILE"
echo "- EchoMemoria: Memory pipeline" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Function to process directory
process_directory() {
    local dir=$1
    local prefix=$2
    
    # Find all .rs files
    find "$dir" -name "*.rs" -type f | sort | while read -r file; do
        echo "Processing: $file"
        
        # Get relative path
        rel_path="${file#$prefix}"
        
        # Skip backup files and disabled files
        if [[ "$file" == *.backup ]] || [[ "$file" == *.disabled ]]; then
            echo "Skipping: $file"
            continue
        fi
        
        # Add file header
        echo "" >> "$OUTPUT_FILE"
        echo "## $rel_path" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "\`\`\`rust" >> "$OUTPUT_FILE"
        
        # Add file content
        cat "$file" >> "$OUTPUT_FILE"
        
        # Close code block
        echo "\`\`\`" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "---" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    done
}

# Process niodoo_real_integrated
echo "Processing niodoo_real_integrated..."
process_directory "niodoo_real_integrated/src" "niodoo_real_integrated/"
if [ -d "niodoo_real_integrated/tests" ]; then
    process_directory "niodoo_real_integrated/tests" "niodoo_real_integrated/"
fi

# Process niodoo-core
echo "Processing niodoo-core..."
process_directory "niodoo-core/src" "niodoo-core/"

# Process constants_core
echo "Processing constants_core..."
process_directory "constants_core/src" "constants_core/"

# Process niodoo-tcs-bridge
echo "Processing niodoo-tcs-bridge..."
process_directory "niodoo-tcs-bridge/src" "niodoo-tcs-bridge/"
if [ -d "niodoo-tcs-bridge/examples" ]; then
    process_directory "niodoo-tcs-bridge/examples" "niodoo-tcs-bridge/"
fi

# Process TCS core modules
echo "Processing TCS modules..."
for tcs_dir in tcs-ml tcs-core tcs-tda tcs-knot tcs-tqft tcs-consensus tcs-pipeline; do
    if [ -d "$tcs_dir/src" ]; then
        echo "Processing $tcs_dir..."
        process_directory "$tcs_dir/src" "$tcs_dir/"
    fi
    if [ -d "$tcs_dir/examples" ]; then
        echo "Processing $tcs_dir examples..."
        process_directory "$tcs_dir/examples" "$tcs_dir/"
    fi
done

# Process src (legacy monolithic)
echo "Processing src (legacy)..."
process_directory "src" "src/"

# Process curator_executor
echo "Processing curator_executor..."
process_directory "curator_executor/src" "curator_executor/"
if [ -d "curator_executor/tests" ]; then
    process_directory "curator_executor/tests" "curator_executor/"
fi

# Process bullshitdetector
echo "Processing bullshitdetector..."
process_directory "bullshitdetector/src" "bullshitdetector/"

# Process EchoMemoria Rust components
echo "Processing EchoMemoria..."
if [ -d "EchoMemoria/src" ]; then
    process_directory "EchoMemoria/src" "EchoMemoria/"
fi

echo ""
echo "Done! Output written to: $OUTPUT_FILE"
echo "File size: $(wc -l < "$OUTPUT_FILE") lines"

