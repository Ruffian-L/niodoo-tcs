#!/usr/bin/env python3
"""Process all BigQuery samples through TDA pipeline"""

import json
import sys
sys.path.insert(0, 'tcs-parser')
from full_pipeline import run_pipeline

def process_all_samples(input_file, output_file):
    """Process all BigQuery samples"""
    processed = 0
    with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
        for line in inf:
            sample = json.loads(line)
            try:
                result = run_pipeline(sample['code_string'], 'rust')
                result['file_id'] = sample['file_id']
                result['path'] = sample['path']
                result['churn_count'] = sample['churn_count']
                result['commit_count'] = sample['commit_count']
                outf.write(json.dumps(result) + '\n')
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} samples...")
            except Exception as e:
                print(f"Error processing {sample['file_id']}: {e}")

    print(f"✓ Processed {processed} samples")

if __name__ == "__main__":
    process_all_samples(
        '/workspace/Niodoo-Final/niodoo-ai/data/rust_topology/rust_bigquery_raw.jsonl',
        '/workspace/Niodoo-Final/niodoo-ai/data/rust_topology/rust_topology_results.jsonl'
    )
