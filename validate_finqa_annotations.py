#!/usr/bin/env python3
"""
FinQA Dataset Annotation Validator

This script validates that program operations in the FinQA dataset match their results.
Example issue: operation labeled as "divide1-1" when it's actually subtraction.
"""

import json
import sys
from typing import List, Dict, Tuple

def validate_operation(op: str, arg1: float, arg2: float, res: float, tolerance: float = 0.01) -> Tuple[bool, str, float]:
    """
    Validate that an operation matches its result.
    
    Args:
        op: Operation name (e.g., "add", "subtract", "divide1-1")
        arg1: First argument
        arg2: Second argument
        res: Claimed result
        tolerance: Tolerance for floating point comparison
        
    Returns:
        (is_valid, correct_operation, expected_result)
    """
    # Try all operations and see which one matches
    operations = {}
    
    # Safe operations
    operations['add'] = arg1 + arg2
    operations['subtract'] = arg1 - arg2
    operations['multiply'] = arg1 * arg2
    
    # Division (check for zero)
    if arg2 != 0:
        operations['divide'] = arg1 / arg2
    else:
        operations['divide'] = None
    
    # Exponentiation (check for overflow)
    try:
        if abs(arg2) < 100:  # Reasonable exponent
            exp_result = arg1 ** arg2
            if abs(exp_result) < 1e10:  # Reasonable result
                operations['exp'] = exp_result
            else:
                operations['exp'] = None
        else:
            operations['exp'] = None
    except (OverflowError, ValueError):
        operations['exp'] = None
    
    # Greater comparison
    operations['greater'] = 1.0 if arg1 > arg2 else 0.0
    
    # Find which operation actually produces the result
    matching_ops = []
    for op_name, expected in operations.items():
        if expected is not None and abs(expected - res) < tolerance:
            matching_ops.append((op_name, expected))
    
    # Check if the claimed operation is correct
    claimed_op_base = op.split('-')[0] if '-' in op else op
    claimed_op_base = claimed_op_base.replace('0', '').replace('1', '').replace('2', '').replace('3', '')
    
    if not matching_ops:
        return False, "NONE", None
    
    correct_op, expected_res = matching_ops[0]
    
    # Check if claimed operation matches
    if claimed_op_base == correct_op:
        return True, correct_op, expected_res
    else:
        return False, correct_op, expected_res


def validate_program(program: List[Dict], tolerance: float = 0.01) -> List[Dict]:
    """Validate all operations in a program."""
    errors = []
    
    for i, step in enumerate(program):
        op = step.get('op', '')
        arg1_str = step.get('arg1', '0')
        arg2_str = step.get('arg2', '0')
        res_str = step.get('res', '0')
        
        # Skip non-numeric arguments (like '#0', 'const_100', etc.)
        if not arg1_str.replace('.', '').replace('-', '').replace('%', '').isdigit():
            continue
        if not arg2_str.replace('.', '').replace('-', '').replace('%', '').isdigit():
            continue
        if not res_str.replace('.', '').replace('-', '').replace('%', '').isdigit():
            continue
            
        try:
            # Handle percentages
            arg1 = float(arg1_str.replace('%', '')) / 100 if '%' in arg1_str else float(arg1_str)
            arg2 = float(arg2_str.replace('%', '')) / 100 if '%' in arg2_str else float(arg2_str)
            res = float(res_str.replace('%', '')) / 100 if '%' in res_str else float(res_str)
            
            is_valid, correct_op, expected_res = validate_operation(op, arg1, arg2, res, tolerance)
            
            if not is_valid:
                errors.append({
                    'step': i,
                    'claimed_op': op,
                    'correct_op': correct_op,
                    'arg1': arg1,
                    'arg2': arg2,
                    'claimed_res': res,
                    'expected_res': expected_res
                })
        except (ValueError, ZeroDivisionError) as e:
            # Skip invalid operations
            pass
    
    return errors


def main():
    """Main validation function."""
    dataset_path = "/root/hsin_research/FinQA-main/dataset/train.json"
    
    print("Loading FinQA train.json...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    print("\nValidating program annotations...\n")
    
    all_errors = []
    
    for entry in data:
        entry_id = entry.get('id', 'unknown')
        steps = entry.get('qa', {}).get('steps', [])
        exe_ans = entry.get('qa', {}).get('exe_ans')
        
        if not steps:
            continue
        
        errors = validate_program(steps)
        
        if errors:
            all_errors.append({
                'id': entry_id,
                'exe_ans': exe_ans,
                'question': entry.get('qa', {}).get('question', ''),
                'program': entry.get('qa', {}).get('program', ''),
                'errors': errors
            })
    
    # Print results
    print("=" * 80)
    print(f"VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nTotal entries checked: {len(data)}")
    print(f"Entries with annotation errors: {len(all_errors)}")
    print(f"Error rate: {len(all_errors) / len(data) * 100:.2f}%\n")
    
    if all_errors:
        print("=" * 80)
        print("DETAILED ERROR REPORT")
        print("=" * 80)
        
        for i, error_entry in enumerate(all_errors[:50], 1):  # Show first 50
            print(f"\n[{i}] ID: {error_entry['id']}")
            print(f"    Expected answer: {error_entry['exe_ans']}")
            print(f"    Program: {error_entry['program']}")
            print(f"    Question: {error_entry['question'][:80]}...")
            print(f"    Errors found:")
            
            for err in error_entry['errors']:
                print(f"      Step {err['step']}:")
                print(f"        Claimed: {err['claimed_op']}({err['arg1']}, {err['arg2']}) = {err['claimed_res']}")
                if err['expected_res'] is not None:
                    print(f"        Correct: {err['correct_op']}({err['arg1']}, {err['arg2']}) = {err['expected_res']:.6f}")
                else:
                    print(f"        Correct: {err['correct_op']} (division by zero or invalid)")
        
        if len(all_errors) > 50:
            print(f"\n... and {len(all_errors) - 50} more entries with errors")
    
    # Save full report
    output_path = "/root/hsin_research/finqa_annotation_errors.json"
    with open(output_path, 'w') as f:
        json.dump(all_errors, f, indent=2)
    
    print(f"\n\nFull error report saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
