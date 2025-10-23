#!/bin/bash

output="all_custom_rust.md"

> $output  # clear the file

find src -type f -name '*.rs' | sort | while read file; do
  echo "## $file" >> $output
  echo '```rust' >> $output
  cat "$file" >> $output
  echo '```' >> $output
  echo "" >> $output
done