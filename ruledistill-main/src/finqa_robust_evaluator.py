"""
FinQA Data Normalization and Robust Evaluator

This module provides:
1. Data Normalization: Converts unit descriptors (million, billion, k) to full numeric values
2. Robust Evaluator: Smart matching with tolerance, ambiguity resolution, and scale invariance

Usage:
    python finqa_robust_evaluator.py --dataset /path/to/train.json --output-log normalization_log.json
"""

import os
import re
import json
import argparse
from typing import Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict


# ============================================================================
# PART 1: DATA NORMALIZATION
# ============================================================================

@dataclass
class NormalizationRecord:
    """Record of a single normalization modification."""
    dataset_id: str
    original_answer: str
    normalized_value: float
    normalization_type: str  # e.g., "million", "billion", "k", "percent"


class DataNormalizer:
    """
    Normalizes FinQA answer values by converting unit descriptors to full numeric form.
    """
    
    # Pattern matchers for different unit types
    UNIT_PATTERNS = {
        'billion': (re.compile(r'([-+]?\d*\.?\d+)\s*billion', re.IGNORECASE), 1e9),
        'million': (re.compile(r'([-+]?\d*\.?\d+)\s*million', re.IGNORECASE), 1e6),
        'thousand': (re.compile(r'([-+]?\d*\.?\d+)\s*thousand', re.IGNORECASE), 1e3),
        'k': (re.compile(r'([-+]?\d*\.?\d+)\s*k\b', re.IGNORECASE), 1e3),
        'm': (re.compile(r'([-+]?\d*\.?\d+)\s*m\b', re.IGNORECASE), 1e6),
        'b': (re.compile(r'([-+]?\d*\.?\d+)\s*b\b', re.IGNORECASE), 1e9),
        'percent': (re.compile(r'([-+]?\d*\.?\d+)\s*%'), 1),  # Keep as-is but note it
    }
    
    def __init__(self):
        self.modification_log = []
        self.stats = defaultdict(int)
    
    def normalize_answer(self, answer: str, dataset_id: str) -> Tuple[str, bool]:
        """
        Normalize an answer string to its full numeric representation.
        
        Args:
            answer: The original answer string
            dataset_id: The ID of the dataset entry
            
        Returns:
            Tuple of (normalized_value_str, was_modified)
        """
        original = answer
        modified = False
        
        # Try each pattern
        for unit_name, (pattern, multiplier) in self.UNIT_PATTERNS.items():
            match = pattern.search(answer)
            if match:
                numeric_part = float(match.group(1))
                
                if unit_name == 'percent':
                    # For percentages, just log but don't change
                    self.stats['percent_values'] += 1
                    continue
                
                # Calculate normalized value
                normalized_value = numeric_part * multiplier
                
                # Record the modification
                record = NormalizationRecord(
                    dataset_id=dataset_id,
                    original_answer=original,
                    normalized_value=normalized_value,
                    normalization_type=unit_name
                )
                self.modification_log.append(record)
                self.stats[f'{unit_name}_converted'] += 1
                
                # Return normalized value as string
                return str(normalized_value), True
        
        return answer, False
    
    def normalize_dataset(self, dataset: list) -> Tuple[list, dict]:
        """
        Normalize all answers in a dataset.
        
        Uses exe_ans (executed answer) as the primary ground truth source,
        falling back to 'answer' only if exe_ans is not available.
        
        Args:
            dataset: List of FinQA data entries
            
        Returns:
            Tuple of (normalized_dataset, summary_stats)
        """
        normalized = []
        exe_ans_count = 0
        answer_fallback_count = 0
        
        for entry in dataset:
            entry_copy = json.loads(json.dumps(entry))  # Deep copy
            
            if 'qa' in entry_copy:
                qa = entry_copy['qa']
                dataset_id = entry_copy.get('id', 'unknown')
                
                # Use exe_ans as primary ground truth (it's the executed program result)
                if 'exe_ans' in qa:
                    original_answer = str(qa['exe_ans'])
                    exe_ans_count += 1
                    source = 'exe_ans'
                elif 'answer' in qa:
                    # Fallback to answer if exe_ans not available
                    original_answer = str(qa['answer'])
                    answer_fallback_count += 1
                    source = 'answer'
                else:
                    normalized.append(entry_copy)
                    continue
                
                normalized_answer, was_modified = self.normalize_answer(
                    original_answer, dataset_id
                )
                
                if was_modified:
                    entry_copy['qa']['original_answer'] = original_answer
                    entry_copy['qa']['answer'] = normalized_answer
                    entry_copy['qa']['answer_source'] = source
            
            normalized.append(entry_copy)
        
        summary = {
            'total_entries': len(dataset),
            'total_modifications': len(self.modification_log),
            'exe_ans_used': exe_ans_count,
            'answer_fallback': answer_fallback_count,
            'by_type': dict(self.stats),
            'modification_rate': len(self.modification_log) / len(dataset) if dataset else 0
        }
        
        return normalized, summary
    
    def get_log(self) -> list:
        """Return the modification log as a list of dicts."""
        return [asdict(record) for record in self.modification_log]


# ============================================================================
# PART 2: ROBUST EVALUATOR
# ============================================================================

class RobustEvaluator:
    """
    Robust evaluator with smart matching for FinQA numerical answers.
    
    Features:
    - Precision tolerance for rounding differences
    - Ambiguity resolution for "how much" questions
    - Scale invariance for factor-of-10 differences
    """
    
    def __init__(
        self,
        tolerance: float = 0.1,
        relative_tolerance: float = 0.01,
        check_scale_factors: bool = True,
        check_ambiguity: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            tolerance: Absolute tolerance for numeric comparison
            relative_tolerance: Relative tolerance (percentage of ground truth)
            check_scale_factors: Check for factor-of-10 matches
            check_ambiguity: Check both absolute and percentage interpretations
        """
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.check_scale_factors = check_scale_factors
        self.check_ambiguity = check_ambiguity
        
        # Patterns to detect ambiguous questions
        self.ambiguity_patterns = [
            re.compile(r'\bhow much\b', re.IGNORECASE),
            re.compile(r'\bwhat was the change\b', re.IGNORECASE),
            re.compile(r'\bby how much\b', re.IGNORECASE),
            re.compile(r'\bwhat is the difference\b', re.IGNORECASE),
            re.compile(r'\bwhat was the increase\b', re.IGNORECASE),
            re.compile(r'\bwhat was the decrease\b', re.IGNORECASE),
        ]
        
        # Scale factors to check (powers of 10)
        self.scale_factors = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    def parse_numeric(self, value: Union[str, float, int]) -> Optional[float]:
        """
        Parse a value to a float, handling various formats.
        
        Args:
            value: The value to parse
            
        Returns:
            Float value or None if unparseable
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if not isinstance(value, str):
            return None
        
        # Clean the string
        cleaned = value.strip()
        
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^[$€£¥]', '', cleaned)
        cleaned = re.sub(r'%$', '', cleaned)
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace(' ', '')
        
        # Handle parentheses for negative numbers
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def is_ambiguous_question(self, question: str) -> bool:
        """Check if the question is ambiguous (could be absolute or percentage)."""
        return any(pattern.search(question) for pattern in self.ambiguity_patterns)
    
    def matches_with_tolerance(
        self,
        predicted: float,
        ground_truth: float
    ) -> Tuple[bool, str]:
        """
        Check if predicted matches ground truth within tolerance.
        
        Returns:
            Tuple of (is_match, match_type)
        """
        if ground_truth == 0:
            if predicted == 0:
                return True, "exact_zero"
            return abs(predicted) < self.tolerance, "near_zero"
        
        # Absolute tolerance
        abs_diff = abs(predicted - ground_truth)
        if abs_diff < self.tolerance:
            return True, "absolute_tolerance"
        
        # Relative tolerance
        rel_diff = abs_diff / abs(ground_truth)
        if rel_diff < self.relative_tolerance:
            return True, "relative_tolerance"
        
        return False, "no_match"
    
    def matches_with_scale(
        self,
        predicted: float,
        ground_truth: float
    ) -> Tuple[bool, str, float]:
        """
        Check if predicted matches ground truth with a scale factor.
        
        Returns:
            Tuple of (is_match, match_type, scale_factor)
        """
        if not self.check_scale_factors:
            return False, "scale_check_disabled", 1
        
        for scale in self.scale_factors:
            scaled_pred = predicted * scale
            is_match, match_type = self.matches_with_tolerance(scaled_pred, ground_truth)
            if is_match:
                return True, f"scale_{scale}_{match_type}", scale
        
        return False, "no_scale_match", 1
    
    def evaluate(
        self,
        predicted: Union[str, float],
        ground_truth: Union[str, float],
        question: str = ""
    ) -> dict:
        """
        Evaluate a single prediction against ground truth.
        
        Args:
            predicted: The predicted answer
            ground_truth: The ground truth answer
            question: The question text (for ambiguity detection)
            
        Returns:
            Dict with evaluation results
        """
        result = {
            'is_correct': False,
            'match_type': 'no_match',
            'predicted_parsed': None,
            'ground_truth_parsed': None,
            'scale_factor': 1,
            'notes': []
        }
        
        # Parse values
        pred_val = self.parse_numeric(predicted)
        gt_val = self.parse_numeric(ground_truth)
        
        result['predicted_parsed'] = pred_val
        result['ground_truth_parsed'] = gt_val
        
        if pred_val is None:
            result['notes'].append('Failed to parse predicted value')
            return result
        
        if gt_val is None:
            result['notes'].append('Failed to parse ground truth value')
            return result
        
        # 1. Direct tolerance match
        is_match, match_type = self.matches_with_tolerance(pred_val, gt_val)
        if is_match:
            result['is_correct'] = True
            result['match_type'] = match_type
            return result
        
        # 2. Scale invariance check
        is_match, match_type, scale = self.matches_with_scale(pred_val, gt_val)
        if is_match:
            result['is_correct'] = True
            result['match_type'] = match_type
            result['scale_factor'] = scale
            result['notes'].append(f'Matched with scale factor {scale}')
            return result
        
        # 3. Ambiguity resolution (for "how much" questions)
        if self.check_ambiguity and self.is_ambiguous_question(question):
            result['notes'].append('Ambiguous question detected')
            
            # Check if prediction matches a percentage interpretation
            # e.g., if gt is 25, check if pred is 0.25 (as 25%)
            if gt_val != 0:
                # Check percentage form
                percentage_gt = gt_val / 100
                is_match, match_type = self.matches_with_tolerance(pred_val, percentage_gt)
                if is_match:
                    result['is_correct'] = True
                    result['match_type'] = f'percentage_interpretation_{match_type}'
                    result['notes'].append('Matched as percentage interpretation')
                    return result
                
                # Check if prediction is the percentage and gt is the absolute
                absolute_from_percent = pred_val * 100
                is_match, match_type = self.matches_with_tolerance(absolute_from_percent, gt_val)
                if is_match:
                    result['is_correct'] = True
                    result['match_type'] = f'absolute_interpretation_{match_type}'
                    result['notes'].append('Matched as absolute interpretation')
                    return result
        
        return result
    
    def evaluate_batch(
        self,
        predictions: list,
        ground_truths: list,
        questions: list = None
    ) -> dict:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted values
            ground_truths: List of ground truth values
            questions: Optional list of questions for ambiguity detection
            
        Returns:
            Dict with batch evaluation results
        """
        if questions is None:
            questions = [''] * len(predictions)
        
        results = []
        correct_count = 0
        match_type_counts = defaultdict(int)
        
        for pred, gt, q in zip(predictions, ground_truths, questions):
            result = self.evaluate(pred, gt, q)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
                match_type_counts[result['match_type']] += 1
        
        return {
            'total': len(predictions),
            'correct': correct_count,
            'accuracy': correct_count / len(predictions) if predictions else 0,
            'match_type_distribution': dict(match_type_counts),
            'individual_results': results
        }


# ============================================================================
# PART 3: PROGRAM EXECUTOR (From FinQA)
# ============================================================================

class ProgramExecutor:
    """
    Executes FinQA programs to compute predicted answers.
    Based on the logic from FinQA's evaluate.py
    """
    
    OPERATIONS = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / b if b != 0 else float('inf'),
        'exp': lambda a, b: a ** b,
        'greater': lambda a, b: 1 if a > b else 0,
    }
    
    def __init__(self, table: list = None):
        """
        Initialize with optional table data.
        
        Args:
            table: List of table rows (each row is a list of cells)
        """
        self.table = table or []
        self.step_results = {}
    
    def parse_value(self, value: str) -> float:
        """Parse a string value to float."""
        if value is None:
            return 0.0
        
        value = str(value).strip()
        
        # Remove common formatting
        value = value.replace(',', '')
        value = value.replace('$', '')
        value = value.replace('%', '')
        value = value.replace(' ', '')
        
        # Handle parentheses (negative)
        if value.startswith('(') and value.endswith(')'):
            value = '-' + value[1:-1]
        
        # Handle special cases
        if value in ['', '-', 'n/a', 'na', 'none', 'null']:
            return 0.0
        
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    def table_lookup(self, row_idx: int, col_idx: int) -> float:
        """Look up a value in the table."""
        try:
            if row_idx < len(self.table) and col_idx < len(self.table[row_idx]):
                return self.parse_value(self.table[row_idx][col_idx])
        except (IndexError, TypeError):
            pass
        return 0.0
    
    def execute_program(self, program: str) -> Tuple[bool, float]:
        """
        Execute a FinQA program string.
        
        Args:
            program: Program string like "subtract(52.4, 32.1)"
            
        Returns:
            Tuple of (success, result)
        """
        try:
            self.step_results = {}
            
            # Parse program into steps
            steps = program.split('),')
            steps = [s.strip() + ')' if not s.strip().endswith(')') else s.strip() 
                     for s in steps if s.strip()]
            
            result = 0.0
            
            for step_idx, step in enumerate(steps):
                # Parse operation and arguments
                match = re.match(r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', step)
                if not match:
                    # Try table operations
                    table_match = re.match(r'table_(\w+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', step)
                    if table_match:
                        op = table_match.group(1)
                        row = int(table_match.group(2))
                        col = int(table_match.group(3))
                        result = self.table_lookup(row, col)
                        self.step_results[step_idx] = result
                        continue
                    return False, 0.0
                
                op_name = match.group(1).lower()
                arg1_str = match.group(2).strip()
                arg2_str = match.group(3).strip()
                
                # Resolve arguments
                arg1 = self._resolve_arg(arg1_str)
                arg2 = self._resolve_arg(arg2_str)
                
                # Execute operation
                if op_name not in self.OPERATIONS:
                    return False, 0.0
                
                result = self.OPERATIONS[op_name](arg1, arg2)
                self.step_results[step_idx] = result
            
            return True, result
            
        except Exception as e:
            return False, 0.0
    
    def _resolve_arg(self, arg: str) -> float:
        """Resolve an argument to a float value."""
        arg = arg.strip()
        
        # Reference to previous step
        if arg.startswith('#'):
            step_idx = int(arg[1:])
            return self.step_results.get(step_idx, 0.0)
        
        # Direct numeric value
        return self.parse_value(arg)


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_normalization(dataset_path: str, output_log_path: str = None) -> dict:
    """
    Run the normalization pass on the dataset.
    
    Args:
        dataset_path: Path to the FinQA dataset JSON
        output_log_path: Optional path to save the modification log
        
    Returns:
        Summary statistics
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} entries")
    
    normalizer = DataNormalizer()
    normalized_dataset, summary = normalizer.normalize_dataset(dataset)
    
    log = normalizer.get_log()
    
    print("\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    print(f"Total entries scanned: {summary['total_entries']}")
    print(f"Total modifications: {summary['total_modifications']}")
    print(f"Modification rate: {summary['modification_rate']*100:.2f}%")
    print("\nBy type:")
    for unit_type, count in summary['by_type'].items():
        print(f"  {unit_type}: {count}")
    
    if output_log_path:
        print(f"\nSaving modification log to {output_log_path}...")
        with open(output_log_path, 'w') as f:
            json.dump({
                'summary': summary,
                'modifications': log
            }, f, indent=2)
    
    # Print sample modifications
    if log:
        print("\n" + "-"*60)
        print("SAMPLE MODIFICATIONS (first 10)")
        print("-"*60)
        for record in log[:10]:
            print(f"  ID: {record['dataset_id']}")
            print(f"    Original: {record['original_answer']}")
            print(f"    Normalized: {record['normalized_value']}")
            print(f"    Type: {record['normalization_type']}")
            print()
    
    return summary


def run_evaluation(dataset_path: str, output_path: str = None) -> dict:
    """
    Run the robust evaluation on the dataset using exe_ans as ground truth.
    
    Args:
        dataset_path: Path to the FinQA dataset JSON
        output_path: Optional path to save evaluation results
        
    Returns:
        Evaluation results
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} entries")
    
    evaluator = RobustEvaluator(
        tolerance=0.1,
        relative_tolerance=0.01,
        check_scale_factors=True,
        check_ambiguity=True
    )
    
    executor = ProgramExecutor()
    
    results = []
    correct_count = 0
    execution_failures = 0
    match_types = defaultdict(int)
    
    for i, entry in enumerate(dataset):
        if 'qa' not in entry:
            continue
        
        qa = entry['qa']
        question = qa.get('question', '')
        program = qa.get('program', '')
        ground_truth = qa.get('exe_ans', qa.get('answer', ''))
        dataset_id = entry.get('id', f'entry_{i}')
        
        # Execute program to get predicted answer
        success, predicted = executor.execute_program(program)
        
        if not success:
            execution_failures += 1
            result = {
                'dataset_id': dataset_id,
                'question': question,
                'program': program,
                'ground_truth': ground_truth,
                'predicted': None,
                'is_correct': False,
                'match_type': 'execution_failure',
                'notes': ['Program execution failed']
            }
        else:
            # Evaluate with robust matching
            eval_result = evaluator.evaluate(predicted, ground_truth, question)
            result = {
                'dataset_id': dataset_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'program': program,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'is_correct': eval_result['is_correct'],
                'match_type': eval_result['match_type'],
                'notes': eval_result['notes']
            }
            
            if eval_result['is_correct']:
                correct_count += 1
            match_types[eval_result['match_type']] += 1
        
        results.append(result)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} entries...")
    
    # Summary
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0
    
    summary = {
        'total_entries': total,
        'correct': correct_count,
        'accuracy': accuracy,
        'execution_failures': execution_failures,
        'match_type_distribution': dict(match_types)
    }
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY (Program Execution → Robust Comparison)")
    print("="*60)
    print(f"Total entries: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Execution failures: {execution_failures}")
    print("\nMatch type distribution:")
    for match_type, count in sorted(match_types.items(), key=lambda x: -x[1]):
        print(f"  {match_type}: {count} ({count/total*100:.1f}%)")
    
    if output_path:
        print(f"\nSaving results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="FinQA Data Normalization and Robust Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/hsin_research/FinQA-main/dataset/train.json",
        help="Path to the FinQA dataset JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/hsin_research/ruledistill-main/data/normalization",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Only run normalization, skip evaluation"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation, skip normalization"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run normalization
    if not args.evaluate_only:
        norm_log_path = os.path.join(args.output_dir, "normalization_log.json")
        run_normalization(args.dataset, norm_log_path)
    
    # Run evaluation
    if not args.normalize_only:
        eval_output_path = os.path.join(args.output_dir, "evaluation_results.json")
        run_evaluation(args.dataset, eval_output_path)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
