#!/usr/bin/env python3
"""

"""

import json
from collections import Counter

# Load errors
with open('/root/hsin_research/finqa_annotation_errors.json') as f:
    errors = json.load(f)

print(f'Total entries with errors: {len(errors)}')
print(f'Total entries in dataset: 6251')
print(f'Error rate: {len(errors)/6251*100:.2f}%\n')

# Count error types
claimed_ops = []
correct_ops = []
for entry in errors:
    for err in entry['errors']:
        claimed_ops.append(err['claimed_op'])
        correct_ops.append(err['correct_op'])

print('Most common INCORRECTLY labeled operations:')
for op, count in Counter(claimed_ops).most_common(15):
    print(f'  {op}: {count}')

print('\nWhat they SHOULD be:')
for op, count in Counter(correct_ops).most_common(10):
    print(f'  {op}: {count}')

# Analyze specific error patterns
minus_to_subtract = sum(1 for c, r in zip(claimed_ops, correct_ops) if 'minus' in c and r == 'subtract')
print(f'\n"minus*" operations that should be "subtract": {minus_to_subtract}')

divide_errors = sum(1 for c in claimed_ops if 'divide' in c and c != 'divide')
print(f'"divide*-*" operations with numeric suffixes: {divide_errors}')

multiply_errors = sum(1 for c in claimed_ops if 'multiply' in c and c != 'multiply')
print(f'"multiply*-*" operations with numeric suffixes: {multiply_errors}')

add_errors = sum(1 for c in claimed_ops if 'add' in c and c != 'add')
print(f'"add*-*" operations with numeric suffixes: {add_errors}')
