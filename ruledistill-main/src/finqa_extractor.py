"""
FinQA Dataset Extractor and Normalizer

Extracts relevant fields from FinQA train.json, normalizes answers,
and evaluates accuracy with/without normalization using FinQA's program execution logic.

Output: /root/hsin_research/data/train.json with extracted fields and normalized answers
"""

import json
import re
import os
from typing import Tuple, Optional, Union
from collections import defaultdict


# ============================================================================
# FINQA PROGRAM EXECUTION (from evaluate.py)
# ============================================================================

ALL_OPS = ["add", "subtract", "multiply", "divide", "exp", "greater", 
           "table_max", "table_min", "table_sum", "table_average"]


def str_to_num(text: str) -> Union[float, str]:
    """Convert string to number, handling various formats."""
    text = str(text).replace(",", "")
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


def process_row(row_in: list) -> Union[list, str]:
    """Process a table row to extract numeric values."""
    row_out = []
    invalid_flag = 0
    
    for num in row_in:
        num = str(num).replace("$", "").strip()
        num = num.split("(")[0].strip()
        
        num = str_to_num(num)
        
        if num == "n/a":
            invalid_flag = 1
            break
        
        row_out.append(num)
    
    if invalid_flag:
        return "n/a"
    
    return row_out


def program_tokenization(original_program: str) -> list:
    """Tokenize a program string into a list of tokens."""
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


def eval_program(program: list, table: list) -> Tuple[int, Union[float, str]]:
    """
    Execute a FinQA program and calculate the numerical result.
    
    Args:
        program: Tokenized program list
        table: Table data as list of rows
        
    Returns:
        Tuple of (invalid_flag, result)
    """
    invalid_flag = 0
    this_res = "n/a"
    
    try:
        program = program[:-1]  # remove EOF
        
        # Check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"
        
        program = "|".join(program)
        steps = program.split(")")[:-1]
        
        res_dict = {}
        
        for ind, step in enumerate(steps):
            step = step.strip()
            
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()
            
            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()
            
            if op in ["add", "subtract", "multiply", "divide", "exp", "greater"]:
                # Resolve arg1
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        invalid_flag = 1
                        break
                
                # Resolve arg2
                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        invalid_flag = 1
                        break
                
                # Execute operation
                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2 if arg2 != 0 else "n/a"
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"
                
                res_dict[ind] = this_res
            
            elif "table" in op:
                table_dict = {}
                for row in table:
                    if row:
                        table_dict[row[0]] = row[1:]
                
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    
                    cal_row = table_dict[arg1]
                    num_row = process_row(cal_row)
                
                if num_row == "n/a":
                    invalid_flag = 1
                    break
                
                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                elif op == "table_average":
                    this_res = sum(num_row) / len(num_row) if num_row else 0
                
                res_dict[ind] = this_res
        
        if this_res != "yes" and this_res != "no" and this_res != "n/a":
            this_res = round(this_res, 5)
    
    except Exception as e:
        invalid_flag = 1
    
    return invalid_flag, this_res


# ============================================================================
# ANSWER NORMALIZATION
# ============================================================================

UNIT_PATTERNS = {
    'billion': (re.compile(r'([-+]?\d*\.?\d+)\s*billion', re.IGNORECASE), 1e9),
    'million': (re.compile(r'([-+]?\d*\.?\d+)\s*million', re.IGNORECASE), 1e6),
    'thousand': (re.compile(r'([-+]?\d*\.?\d+)\s*thousand', re.IGNORECASE), 1e3),
    'k': (re.compile(r'([-+]?\d*\.?\d+)\s*k\b', re.IGNORECASE), 1e3),
}


def normalize_answer(answer: str) -> Tuple[str, bool, str]:
    """
    Normalize an answer by converting unit descriptors to full numeric values.
    
    Returns:
        Tuple of (normalized_value, was_modified, unit_type)
    """
    original = str(answer)
    
    for unit_name, (pattern, multiplier) in UNIT_PATTERNS.items():
        match = pattern.search(original)
        if match:
            numeric_part = float(match.group(1))
            normalized_value = numeric_part * multiplier
            return str(normalized_value), True, unit_name
    
    return original, False, None


def parse_answer_numeric(answer: str) -> Optional[float]:
    """Parse an answer string to a float value."""
    if answer is None:
        return None
    
    answer = str(answer).strip()
    
    # Remove common prefixes/suffixes
    answer = re.sub(r'^[$€£¥]', '', answer)
    answer = re.sub(r'%$', '', answer)
    answer = answer.replace(',', '')
    answer = answer.replace(' ', '')
    
    # Handle parentheses for negative
    if answer.startswith('(') and answer.endswith(')'):
        answer = '-' + answer[1:-1]
    
    try:
        return float(answer)
    except ValueError:
        return None


def compare_answers(predicted, ground_truth, tolerance: float = 0.01) -> bool:
    """
    Compare predicted answer with ground truth using tolerance.
    
    Args:
        predicted: Predicted answer (from program execution)
        ground_truth: Ground truth answer
        tolerance: Relative tolerance for comparison
        
    Returns:
        True if answers match within tolerance
    """
    if predicted == "n/a":
        return False
    
    if isinstance(predicted, str) and predicted in ["yes", "no"]:
        return str(ground_truth).lower().strip() == predicted
    
    try:
        pred_val = float(predicted)
        gt_val = parse_answer_numeric(str(ground_truth))
        
        if gt_val is None:
            return False
        
        if gt_val == 0:
            return abs(pred_val) < tolerance
        
        # Check relative difference
        rel_diff = abs(pred_val - gt_val) / abs(gt_val)
        return rel_diff < tolerance
    
    except (ValueError, TypeError):
        return str(predicted) == str(ground_truth)


# ============================================================================
# MAIN EXTRACTION
# ============================================================================

def extract_and_process_dataset(
    input_path: str,
    output_path: str
) -> dict:
    """
    Extract relevant fields from FinQA dataset and add normalized answers.
    
    Args:
        input_path: Path to original train.json
        output_path: Path to save extracted dataset
        
    Returns:
        Summary statistics
    """
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} entries")
    
    extracted_data = []
    stats = {
        'total': len(dataset),
        'normalized': 0,
        'program_execution_success': 0,
        'program_execution_failed': 0,
        'accuracy_original': 0,
        'accuracy_normalized': 0,
        'normalization_types': defaultdict(int)
    }
    
    correct_original = 0
    correct_normalized = 0
    
    for i, entry in enumerate(dataset):
        if 'qa' not in entry:
            continue
        
        qa = entry['qa']
        table = entry.get('table', [])
        
        # Extract fields
        extracted = {
            'id': entry.get('id', f'entry_{i}'),
            'question': qa.get('question', ''),
            'answer': qa.get('answer', ''),
            'gold_inds': qa.get('gold_inds', {}),
            'program': qa.get('program', ''),
            'exe_ans': qa.get('exe_ans', None),  # Ground truth from program
        }
        
        # Normalize answer
        normalized_answer, was_normalized, unit_type = normalize_answer(extracted['answer'])
        extracted['answer_normalized'] = normalized_answer
        extracted['was_normalized'] = was_normalized
        
        if was_normalized:
            stats['normalized'] += 1
            stats['normalization_types'][unit_type] += 1
        
        # Execute program to get predicted answer
        program_tokens = program_tokenization(qa.get('program', ''))
        invalid_flag, predicted = eval_program(program_tokens, table)
        
        extracted['predicted_from_program'] = predicted
        extracted['program_valid'] = invalid_flag == 0
        
        if invalid_flag == 0:
            stats['program_execution_success'] += 1
        else:
            stats['program_execution_failed'] += 1
        
        # Compare with original answer
        if extracted['exe_ans'] is not None:
            gt = extracted['exe_ans']
        else:
            gt = extracted['answer']
        
        is_correct_original = compare_answers(predicted, gt)
        is_correct_normalized = compare_answers(predicted, normalized_answer)
        
        extracted['correct_vs_original'] = is_correct_original
        extracted['correct_vs_normalized'] = is_correct_normalized
        
        if is_correct_original:
            correct_original += 1
        if is_correct_normalized:
            correct_normalized += 1
        
        extracted_data.append(extracted)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)}...")
    
    stats['accuracy_original'] = correct_original / len(extracted_data) if extracted_data else 0
    stats['accuracy_normalized'] = correct_normalized / len(extracted_data) if extracted_data else 0
    stats['normalization_types'] = dict(stats['normalization_types'])
    
    # Save extracted dataset
    print(f"\nSaving extracted dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION & NORMALIZATION SUMMARY")
    print("="*60)
    print(f"Total entries: {stats['total']}")
    print(f"Entries normalized: {stats['normalized']}")
    print(f"Program execution success: {stats['program_execution_success']}")
    print(f"Program execution failed: {stats['program_execution_failed']}")
    print("\nNormalization by type:")
    for unit_type, count in stats['normalization_types'].items():
        print(f"  {unit_type}: {count}")
    
    print("\n" + "="*60)
    print("ACCURACY COMPARISON")
    print("="*60)
    print(f"Accuracy (vs original answer): {stats['accuracy_original']*100:.2f}%")
    print(f"Accuracy (vs normalized answer): {stats['accuracy_normalized']*100:.2f}%")
    print(f"Difference: {(stats['accuracy_normalized'] - stats['accuracy_original'])*100:+.2f}%")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and normalize FinQA dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="/root/hsin_research/FinQA-main/dataset/train.json",
        help="Input dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/hsin_research/data/train.json",
        help="Output dataset path"
    )
    
    args = parser.parse_args()
    
    stats = extract_and_process_dataset(args.input, args.output)
    
    # Save stats
    stats_path = args.output.replace('.json', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
