"""
FinQA DSL Evaluator

This module implements the FinQA Domain-Specific Language (DSL) program executor.
It can execute programs like: ["subtract(", "6348", "6241", ")", "divide(", "#0", "6241", ")", "EOF"]

Based on the FinQA evaluation system from: https://github.com/czyssrs/FinQA
"""

import re
from typing import Tuple, List, Union, Optional


# All supported operations in FinQA DSL
ALL_OPS = [
    "add", "subtract", "multiply", "divide", "exp", "greater",
    "table_max", "table_min", "table_sum", "table_average"
]


def str_to_num(text: str) -> Union[float, str]:
    """
    Convert string to number, handling various formats.
    
    Supports:
    - Regular numbers: "123.4"
    - Percentages: "50%" -> 0.5
    - Constants: "const_100" -> 100.0, "const_m1" -> -1.0
    - Comma-separated: "1,234" -> 1234.0
    - Currency symbols: "$100" -> 100.0
    
    Returns:
        float value or "n/a" if parsing fails
    """
    text = str(text).strip()
    
    # Handle percentages
    if "%" in text:
        text = text.replace("%", "")
        try:
            num = float(text)
            return num / 100.0
        except ValueError:
            return "n/a"
    
    # Handle constants
    elif "const" in text:
        text = text.replace("const_", "")
        if text == "m1":
            text = "-1"
        try:
            return float(text)
        except ValueError:
            return "n/a"
    
    else:
        # Clean formatting
        text = text.replace(",", "")
        text = text.replace("$", "")
        text = text.replace("€", "")
        text = text.replace("£", "")
        text = text.strip()
        
        # Handle parentheses for negative numbers
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]
        
        try:
            return float(text)
        except ValueError:
            return "n/a"


def process_row(row: List[str]) -> Union[List[float], str]:
    """
    Process a table row, converting all cells to numbers.
    
    Args:
        row: List of cell values
        
    Returns:
        List of float values or "n/a" if any cell fails to parse
    """
    num_row = []
    for cell in row:
        num = str_to_num(cell)
        if num == "n/a":
            return "n/a"
        num_row.append(num)
    return num_row


def eval_program(program: List[str], table: List[List[str]] = None) -> Tuple[int, Union[float, str]]:
    """
    Execute a FinQA DSL program.
    
    Args:
        program: List of tokens like ["subtract(", "6348", "6241", ")", "EOF"]
        table: Optional table data as list of rows (each row is list of cells)
        
    Returns:
        Tuple of (invalid_flag, result)
        - invalid_flag: 0 if successful, 1 if error
        - result: computed value or "n/a" if error
    """
    if table is None:
        table = []
    
    invalid_flag = 0
    this_res = "n/a"
    
    try:
        # Remove EOF
        if program and program[-1] == "EOF":
            program = program[:-1]
        
        # Check structure: every 4 tokens should be (op, arg1, arg2, ")")
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"
        
        # Parse into steps
        program_str = "|".join(program)
        steps = program_str.split(")")[:-1]
        
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
            
            # Arithmetic operations
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
                    if arg2 == 0:
                        invalid_flag = 1
                        break
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                elif op == "greater":
                    this_res = "yes" if arg1 > arg2 else "no"
                
                res_dict[ind] = this_res
            
            # Table operations
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
                    this_res = sum(num_row) / len(num_row)
                
                res_dict[ind] = this_res
        
        # Round numeric results
        if this_res not in ["yes", "no", "n/a"]:
            this_res = round(this_res, 5)
    
    except Exception as e:
        invalid_flag = 1
    
    return invalid_flag, this_res


def parse_program_from_string(program_str: str) -> List[str]:
    """
    Parse a program string into token list.
    
    Accepts formats like:
    - '["subtract(", "6348", "6241", ")", "EOF"]'
    - 'subtract(6348, 6241)'
    - 'subtract(6348, 6241), divide(#0, 100)'
    
    Returns:
        List of tokens
    """
    program_str = program_str.strip()
    
    # Try JSON format first
    if program_str.startswith("["):
        try:
            import json
            return json.loads(program_str)
        except:
            pass
    
    # Parse function-call format
    tokens = []
    
    # Match pattern: operation(arg1, arg2)
    pattern = r'(\w+)\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)'
    
    for match in re.finditer(pattern, program_str):
        op = match.group(1)
        arg1 = match.group(2).strip()
        arg2 = match.group(3).strip()
        
        tokens.extend([f"{op}(", arg1, arg2, ")"])
    
    if tokens:
        tokens.append("EOF")
    
    return tokens


def compare_results(predicted: Union[float, str], ground_truth: Union[float, str], 
                   tolerance: float = 0.01) -> bool:
    """
    Compare predicted result with ground truth.
    
    Args:
        predicted: Predicted value
        ground_truth: Ground truth value
        tolerance: Relative tolerance for numeric comparison
        
    Returns:
        True if results match within tolerance
    """
    # Handle string comparisons (yes/no)
    if isinstance(predicted, str) and isinstance(ground_truth, str):
        return predicted.lower().strip() == ground_truth.lower().strip()
    
    # Handle n/a
    if predicted == "n/a" or ground_truth == "n/a":
        return False
    
    # Numeric comparison
    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
        
        if gt_val == 0:
            return abs(pred_val) < tolerance
        
        rel_diff = abs(pred_val - gt_val) / abs(gt_val)
        return rel_diff < tolerance
    
    except (ValueError, TypeError):
        return str(predicted) == str(ground_truth)


class DSLEvaluator:
    """
    Evaluator for FinQA DSL programs.
    
    This class wraps the program execution and result comparison logic.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize the evaluator.
        
        Args:
            tolerance: Relative tolerance for numeric comparison
        """
        self.tolerance = tolerance
        self.stats = {
            "total": 0,
            "correct": 0,
            "execution_errors": 0,
            "comparison_errors": 0
        }
    
    def evaluate(self, program: Union[str, List[str]], ground_truth: Union[float, str],
                table: List[List[str]] = None) -> dict:
        """
        Evaluate a single program.
        
        Args:
            program: DSL program (string or token list)
            ground_truth: Expected result
            table: Optional table data
            
        Returns:
            Dictionary with evaluation results
        """
        self.stats["total"] += 1
        
        # Parse program if string
        if isinstance(program, str):
            program = parse_program_from_string(program)
        
        # Execute program
        invalid_flag, result = eval_program(program, table)
        
        if invalid_flag:
            self.stats["execution_errors"] += 1
            return {
                "success": False,
                "error_type": "execution_error",
                "program": program,
                "result": result,
                "ground_truth": ground_truth,
                "is_correct": False
            }
        
        # Compare results
        is_correct = compare_results(result, ground_truth, self.tolerance)
        
        if is_correct:
            self.stats["correct"] += 1
        else:
            self.stats["comparison_errors"] += 1
        
        return {
            "success": True,
            "program": program,
            "result": result,
            "ground_truth": ground_truth,
            "is_correct": is_correct
        }
    
    def evaluate_batch(self, programs: List[Union[str, List[str]]], 
                      ground_truths: List[Union[float, str]],
                      tables: List[List[List[str]]] = None) -> dict:
        """
        Evaluate a batch of programs.
        
        Args:
            programs: List of DSL programs
            ground_truths: List of ground truth values
            tables: Optional list of tables
            
        Returns:
            Dictionary with batch evaluation results
        """
        if tables is None:
            tables = [None] * len(programs)
        
        results = []
        for program, gt, table in zip(programs, ground_truths, tables):
            result = self.evaluate(program, gt, table)
            results.append(result)
        
        return {
            "total": self.stats["total"],
            "correct": self.stats["correct"],
            "accuracy": self.stats["correct"] / self.stats["total"] if self.stats["total"] > 0 else 0,
            "execution_errors": self.stats["execution_errors"],
            "comparison_errors": self.stats["comparison_errors"],
            "results": results
        }
    
    def reset_stats(self):
        """Reset evaluation statistics."""
        self.stats = {
            "total": 0,
            "correct": 0,
            "execution_errors": 0,
            "comparison_errors": 0
        }
    
    def evaluate_with_gt_program(
        self, 
        model_program: Union[str, List[str]], 
        gt_program: str,
        table: List[List[str]] = None
    ) -> dict:
        """
        Evaluate by executing both model's program and ground truth program.
        
        This compares the execution results of both programs rather than
        comparing to a pre-computed exe_ans value.
        
        Args:
            model_program: Model's generated DSL program
            gt_program: Ground truth program string from dataset (e.g., "divide(100, 50)")
            table: Optional table data
            
        Returns:
            Dictionary with evaluation results including both program executions
        """
        self.stats["total"] += 1
        
        # Parse model program if string
        if isinstance(model_program, str):
            model_tokens = parse_program_from_string(model_program)
        else:
            model_tokens = model_program
        
        # Parse ground truth program
        gt_tokens = parse_program_from_string(gt_program)
        
        # Execute ground truth program
        gt_invalid, gt_result = eval_program(gt_tokens, table)
        
        if gt_invalid:
            # Ground truth program failed - this shouldn't happen
            return {
                "success": False,
                "error_type": "gt_execution_error",
                "model_program": model_tokens,
                "gt_program": gt_tokens,
                "model_result": None,
                "gt_result": None,
                "is_correct": False,
                "program_match": False
            }
        
        # Execute model program
        model_invalid, model_result = eval_program(model_tokens, table)
        
        if model_invalid:
            self.stats["execution_errors"] += 1
            return {
                "success": False,
                "error_type": "model_execution_error",
                "model_program": model_tokens,
                "gt_program": gt_tokens,
                "model_result": None,
                "gt_result": gt_result,
                "is_correct": False,
                "program_match": False
            }
        
        # Compare execution results
        results_match = compare_results(model_result, gt_result, self.tolerance)
        
        # Check if programs are exactly the same
        program_match = self._compare_programs(model_tokens, gt_tokens)
        
        if results_match:
            self.stats["correct"] += 1
        else:
            self.stats["comparison_errors"] += 1
        
        return {
            "success": True,
            "model_program": model_tokens,
            "gt_program": gt_tokens,
            "model_result": model_result,
            "gt_result": gt_result,
            "is_correct": results_match,
            "program_match": program_match
        }
    
    def _compare_programs(self, prog1: List[str], prog2: List[str]) -> bool:
        """
        Check if two programs are exactly the same.
        
        Args:
            prog1: First program tokens
            prog2: Second program tokens
            
        Returns:
            True if programs match exactly
        """
        if not prog1 or not prog2:
            return False
        
        # Remove EOF for comparison
        p1 = [t for t in prog1 if t != "EOF"]
        p2 = [t for t in prog2 if t != "EOF"]
        
        if len(p1) != len(p2):
            return False
        
        for t1, t2 in zip(p1, p2):
            # Normalize tokens for comparison
            t1_norm = str(t1).strip().lower()
            t2_norm = str(t2).strip().lower()
            
            # Try numeric comparison for arguments
            try:
                if abs(float(t1_norm.rstrip('%')) - float(t2_norm.rstrip('%'))) < 0.0001:
                    continue
            except ValueError:
                pass
            
            if t1_norm != t2_norm:
                return False
        
        return True
    
    def evaluate_batch_with_gt_programs(
        self,
        model_programs: List[Union[str, List[str]]],
        gt_programs: List[str],
        tables: List[List[List[str]]] = None
    ) -> dict:
        """
        Evaluate a batch by comparing model and ground truth program executions.
        
        Args:
            model_programs: List of model-generated programs
            gt_programs: List of ground truth program strings
            tables: Optional list of tables
            
        Returns:
            Dictionary with batch evaluation results
        """
        if tables is None:
            tables = [None] * len(model_programs)
        
        results = []
        program_matches = 0
        
        for model_prog, gt_prog, table in zip(model_programs, gt_programs, tables):
            result = self.evaluate_with_gt_program(model_prog, gt_prog, table)
            results.append(result)
            
            if result.get("program_match", False):
                program_matches += 1
        
        return {
            "total": self.stats["total"],
            "correct": self.stats["correct"],
            "execution_accuracy": self.stats["correct"] / self.stats["total"] if self.stats["total"] > 0 else 0,
            "program_match_count": program_matches,
            "program_match_rate": program_matches / len(model_programs) if model_programs else 0,
            "execution_errors": self.stats["execution_errors"],
            "comparison_errors": self.stats["comparison_errors"],
            "results": results
        }


# Test the evaluator
if __name__ == "__main__":
    print("Testing DSL Evaluator...")
    print("=" * 60)
    
    evaluator = DSLEvaluator()
    
    # Test cases
    test_cases = [
        {
            "program": ["subtract(", "6348", "6241", ")", "divide(", "#0", "6241", ")", "EOF"],
            "ground_truth": 0.01715,
            "description": "Percentage change calculation"
        },
        {
            "program": ["add(", "100", "200", ")", "EOF"],
            "ground_truth": 300,
            "description": "Simple addition"
        },
        {
            "program": ["multiply(", "10", "5", ")", "subtract(", "#0", "25", ")", "EOF"],
            "ground_truth": 25,
            "description": "Multiply then subtract"
        },
        {
            "program": ["divide(", "50%", "2", ")", "EOF"],
            "ground_truth": 0.25,
            "description": "Percentage handling"
        },
        {
            "program": ["greater(", "100", "50", ")", "EOF"],
            "ground_truth": "yes",
            "description": "Comparison operation"
        },
    ]
    
    for tc in test_cases:
        result = evaluator.evaluate(tc["program"], tc["ground_truth"])
        status = "✅" if result["is_correct"] else "❌"
        print(f"{status} {tc['description']}")
        print(f"   Program: {tc['program']}")
        print(f"   Result: {result['result']} | Expected: {tc['ground_truth']}")
        print()
    
    print("=" * 60)
    print(f"Accuracy: {evaluator.stats['correct']}/{evaluator.stats['total']} "
          f"({evaluator.stats['correct']/evaluator.stats['total']*100:.1f}%)")
