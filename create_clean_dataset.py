#!/usr/bin/env python3
"""
Create a clean version of train.json by removing all entries with annotation errors.
"""

import json

# Load the original train.json from before corruption
print("Loading original error report (with all 2,571 errors including minus/subtract issues)...")

# We need to use the original validation output to get all 2,571 error IDs
# Let me load the full error report that includes minus/subtract issues
with open('/root/hsin_research/finqa_real_errors.json') as f:
    error_entries = json.load(f)

# Get the set of error IDs from the FILTERED report (430 entries)
# But the user wants to exclude the original 2,571, so we need to re-run validation
# to get all error IDs

# Actually, let me check if we have the full list somewhere
# For now, I'll use the filtered errors as a starting point
error_ids = set()
for entry in error_entries:
    error_ids.add(entry['id'])

print(f"Found {len(error_ids)} error IDs in filtered report")
print(f"Note: This is the FILTERED count (430). Will re-validate to get all 2,571 errors.")

# Load and re-validate train.json to get ALL error IDs
print("\nRe-validating train.json to find all errors (including minus/subtract)...")

# Import validation functions
import sys
sys.path.insert(0, '/root/hsin_research')

# Re-run validation to get ALL errors
from validate_finqa_annotations import validate_program

# Try loading from dev.json since train.json is corrupted
try:
    # Actually, let's check if we can reconstruct from a backup
    print("Attempting to load train.json...")
    with open('/root/hsin_research/FinQA-main/dataset/train.json') as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} entries from train.json")
except Exception as e:
    print(f"Error loading train.json: {e}")
    print("train.json appears to be corrupted. Cannot create clean dataset.")
    print("Please restore train.json from backup first.")
    exit(1)

# Validate all entries to find ALL errors (not just filtered ones)
all_error_ids = set()
for entry in data:
    entry_id = entry.get('id', 'unknown')
    steps = entry.get('qa', {}).get('steps', [])
    
    if not steps:
        continue
    
    errors = validate_program(steps, tolerance=0.01)
    
    if errors:
        all_error_ids.add(entry_id)

print(f"\nTotal entries with ANY annotation errors: {len(all_error_ids)}")

# Filter out all error entries
clean_data = []
for entry in data:
    if entry.get('id') not in all_error_ids:
        clean_data.append(entry)

print(f"\nOriginal entries: {len(data)}")
print(f"Entries with errors: {len(all_error_ids)}")
print(f"Clean entries: {len(clean_data)}")

# Save clean dataset
output_path = '/root/hsin_research/FinQA-main/dataset/train-clear.json'
with open(output_path, 'w') as f:
    json.dump(clean_data, f, indent=2)

print(f"\n✅ Clean dataset saved to: {output_path}")
print(f"   Contains {len(clean_data)} entries ({len(clean_data)/len(data)*100:.2f}% of original)")
