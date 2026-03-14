#!/usr/bin/env python3
"""
Filter finqa_annotation_errors.json to remove minus/subtract naming inconsistencies.
Keep only entries with actual calculation errors.
"""

import json

# Load the errors
with open('/root/hsin_research/finqa_annotation_errors.json') as f:
    all_errors = json.load(f)

print(f"Original entries with errors: {len(all_errors)}")

# Filter out entries where ALL errors are just minus→subtract naming issues
filtered_errors = []

for entry in all_errors:
    # Check if any error is NOT just a minus→subtract naming issue
    has_real_error = False
    
    for err in entry['errors']:
        claimed_op = err['claimed_op']
        correct_op = err['correct_op']
        
        # Check if this is just a minus→subtract naming issue
        is_minus_subtract = ('minus' in claimed_op and correct_op == 'subtract')
        
        if not is_minus_subtract:
            # This is a real error (not just naming)
            has_real_error = True
            break
    
    # Only keep entries with at least one real error
    if has_real_error:
        # Filter out the minus→subtract errors from this entry too
        real_errors = [
            err for err in entry['errors']
            if not ('minus' in err['claimed_op'] and err['correct_op'] == 'subtract')
        ]
        
        if real_errors:  # Only add if there are remaining errors
            entry['errors'] = real_errors
            filtered_errors.append(entry)

print(f"Filtered entries (real errors only): {len(filtered_errors)}")
print(f"Removed: {len(all_errors) - len(filtered_errors)} entries")

# Count error types in filtered set
from collections import Counter
claimed_ops = []
for entry in filtered_errors:
    for err in entry['errors']:
        claimed_ops.append(err['claimed_op'])

print("\nRemaining error types:")
for op, count in Counter(claimed_ops).most_common(10):
    print(f"  {op}: {count}")

# Save filtered errors
output_path = '/root/hsin_research/finqa_real_errors.json'
with open(output_path, 'w') as f:
    json.dump(filtered_errors, f, indent=2)

print(f"\n✅ Filtered errors saved to: {output_path}")
