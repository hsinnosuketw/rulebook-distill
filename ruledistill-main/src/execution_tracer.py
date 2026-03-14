"""
Execution Tracer Module

Provides step-by-step execution tracing for DSL programs with debugging support.
This module EXTENDS the existing dsl_evaluator.py functionality.

Key Features:
- ExecutionTrace dataclass for storing step-by-step results
- eval_program_with_trace(): returns full execution trace for debugging
- TraceComparator: diffs model vs GT traces to generate diagnoses

Based on CodePRM research for Process Reward Model training.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Literal
import re

# Import existing DSL operations from dsl_evaluator
from dsl_evaluator import ALL_OPS, str_to_num, process_row


@dataclass
class ExecutionStep:
    """Single step in DSL program execution."""
    step_index: int
    operation: str
    arg1_raw: str  # Original string argument
    arg2_raw: str  # Original string argument  
    arg1_resolved: Union[float, str]  # Resolved value
    arg2_resolved: Union[float, str]  # Resolved value
    result: Union[float, str]
    status: Literal["OK", "ERROR", "SKIPPED"] = "OK"
    error_message: str = ""


@dataclass
class ExecutionTrace:
    """Complete execution trace for a DSL program."""
    program: List[str]
    steps: List[ExecutionStep] = field(default_factory=list)
    final_result: Union[float, str] = "n/a"
    is_valid: bool = False
    error_message: str = ""
    
    def to_string(self, prefix: str = "") -> str:
        """Format trace as human-readable string for optimizer feedback."""
        lines = [f"{prefix}EXECUTION TRACE:"]
        for step in self.steps:
            status_icon = "✓" if step.status == "OK" else "✗"
            lines.append(
                f"{prefix}  {step.step_index}. {step.operation}({step.arg1_raw}, {step.arg2_raw}) "
                f"-> {step.result}  [{status_icon} {step.status}]"
            )
            if step.error_message:
                lines.append(f"{prefix}     Error: {step.error_message}")
        lines.append(f"{prefix}Final Result: {self.final_result}")
        return "\n".join(lines)


def eval_program_with_trace(
    program: List[str], 
    table: List[List[str]] = None
) -> ExecutionTrace:
    """
    Execute a FinQA DSL program with full execution tracing.
    
    This EXTENDS eval_program() by returning step-by-step traces
    for debugging and optimizer feedback.
    
    Args:
        program: List of tokens like ["subtract(", "6348", "6241", ")", "EOF"]
        table: Optional table data
        
    Returns:
        ExecutionTrace with step-by-step execution details
    """
    if table is None:
        table = []
    
    trace = ExecutionTrace(program=program.copy())
    
    try:
        # Remove EOF
        if program and program[-1] == "EOF":
            program = program[:-1]
        
        # Check structure: every 4 tokens should be (op, arg1, arg2, ")")
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in ALL_OPS:
                    trace.error_message = f"Invalid operation: {token}"
                    return trace
            if (ind + 1) % 4 == 0:
                if token != ")":
                    trace.error_message = f"Expected ')' at position {ind}"
                    return trace
        
        # Parse into steps
        program_str = "|".join(program)
        steps = program_str.split(")")[:-1]
        
        res_dict = {}
        
        for ind, step_str in enumerate(steps):
            step_str = step_str.strip()
            
            if len(step_str.split("(")) > 2:
                trace.error_message = f"Malformed step: {step_str}"
                return trace
            
            op = step_str.split("(")[0].strip("|").strip()
            args = step_str.split("(")[1].strip("|").strip()
            
            arg1_raw = args.split("|")[0].strip()
            arg2_raw = args.split("|")[1].strip()
            
            step = ExecutionStep(
                step_index=ind,
                operation=op,
                arg1_raw=arg1_raw,
                arg2_raw=arg2_raw,
                arg1_resolved="n/a",
                arg2_resolved="n/a",
                result="n/a"
            )
            
            # Arithmetic operations
            if op in ["add", "subtract", "multiply", "divide", "exp", "greater"]:
                # Resolve arg1
                if "#" in arg1_raw:
                    ref_idx = int(arg1_raw.replace("#", ""))
                    arg1 = res_dict.get(ref_idx, "n/a")
                else:
                    arg1 = str_to_num(arg1_raw)
                
                step.arg1_resolved = arg1
                
                if arg1 == "n/a":
                    step.status = "ERROR"
                    step.error_message = f"Could not resolve arg1: {arg1_raw}"
                    step.result = "n/a"
                    trace.steps.append(step)
                    trace.error_message = step.error_message
                    return trace
                
                # Resolve arg2
                if "#" in arg2_raw:
                    ref_idx = int(arg2_raw.replace("#", ""))
                    arg2 = res_dict.get(ref_idx, "n/a")
                else:
                    arg2 = str_to_num(arg2_raw)
                
                step.arg2_resolved = arg2
                
                if arg2 == "n/a":
                    step.status = "ERROR"
                    step.error_message = f"Could not resolve arg2: {arg2_raw}"
                    step.result = "n/a"
                    trace.steps.append(step)
                    trace.error_message = step.error_message
                    return trace
                
                # Execute operation
                try:
                    if op == "add":
                        result = arg1 + arg2
                    elif op == "subtract":
                        result = arg1 - arg2
                    elif op == "multiply":
                        result = arg1 * arg2
                    elif op == "divide":
                        if arg2 == 0:
                            step.status = "ERROR"
                            step.error_message = "Division by zero"
                            step.result = "n/a"
                            trace.steps.append(step)
                            trace.error_message = step.error_message
                            return trace
                        result = arg1 / arg2
                    elif op == "exp":
                        result = arg1 ** arg2
                    elif op == "greater":
                        result = "yes" if arg1 > arg2 else "no"
                    
                    step.result = result
                    step.status = "OK"
                    res_dict[ind] = result
                    
                except Exception as e:
                    step.status = "ERROR"
                    step.error_message = str(e)
                    step.result = "n/a"
                    trace.steps.append(step)
                    trace.error_message = step.error_message
                    return trace
            
            # Table operations
            elif "table" in op:
                table_dict = {}
                for row in table:
                    if row:
                        table_dict[row[0]] = row[1:]
                
                if "#" in arg1_raw:
                    ref_idx = int(arg1_raw.replace("#", ""))
                    arg1 = res_dict.get(ref_idx, "n/a")
                    step.arg1_resolved = arg1
                else:
                    if arg1_raw not in table_dict:
                        step.status = "ERROR"
                        step.error_message = f"Row not found in table: {arg1_raw}"
                        step.result = "n/a"
                        trace.steps.append(step)
                        trace.error_message = step.error_message
                        return trace
                    
                    cal_row = table_dict[arg1_raw]
                    num_row = process_row(cal_row)
                    step.arg1_resolved = f"row:{len(cal_row)} values"
                    
                    if num_row == "n/a":
                        step.status = "ERROR"
                        step.error_message = f"Could not parse row values"
                        step.result = "n/a"
                        trace.steps.append(step)
                        trace.error_message = step.error_message
                        return trace
                
                    try:
                        if op == "table_max":
                            result = max(num_row)
                        elif op == "table_min":
                            result = min(num_row)
                        elif op == "table_sum":
                            result = sum(num_row)
                        elif op == "table_average":
                            result = sum(num_row) / len(num_row)
                        
                        step.result = result
                        step.status = "OK"
                        res_dict[ind] = result
                    except Exception as e:
                        step.status = "ERROR"
                        step.error_message = str(e)
                        step.result = "n/a"
                        trace.steps.append(step)
                        trace.error_message = step.error_message
                        return trace
            
            trace.steps.append(step)
        
        # Get final result
        if trace.steps:
            final = trace.steps[-1].result
            if final not in ["yes", "no", "n/a"] and isinstance(final, (int, float)):
                final = round(final, 5)
            trace.final_result = final
            trace.is_valid = True
        
    except Exception as e:
        trace.error_message = str(e)
        trace.is_valid = False
    
    return trace


class TraceComparator:
    """
    Compares model vs ground truth execution traces to generate diagnoses.
    
    This implements the CodePRM-style feedback where we identify
    WHERE in the execution the model diverged from the correct solution.
    """
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
    
    def compare_values(self, v1: Union[float, str], v2: Union[float, str]) -> bool:
        """Check if two values match within tolerance."""
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.lower().strip() == v2.lower().strip()
        
        try:
            f1, f2 = float(v1), float(v2)
            if f2 == 0:
                return abs(f1) < self.tolerance
            return abs(f1 - f2) / abs(f2) < self.tolerance
        except (ValueError, TypeError):
            return str(v1) == str(v2)
    
    def diff_traces(
        self, 
        model_trace: ExecutionTrace, 
        gt_trace: ExecutionTrace,
        question: str = ""
    ) -> dict:
        """
        Generate diagnostic diff between model and GT traces.
        
        Returns:
            Dictionary with:
            - diagnosis: str (human-readable explanation)
            - divergence_point: int (step index where traces diverged)
            - error_type: str (STRUCTURE, OPERAND, OPERATION, EXECUTION)
            - corrective_hint: str (suggestion for rule improvement)
        """
        diagnosis_parts = []
        divergence_point = -1
        error_type = "UNKNOWN"
        corrective_hint = ""
        
        # Check if model failed to execute
        if not model_trace.is_valid:
            return {
                "diagnosis": f"EXECUTION ERROR: {model_trace.error_message}",
                "divergence_point": 0,
                "error_type": "EXECUTION",
                "corrective_hint": "Model program syntax is invalid. Check DSL format.",
                "model_trace": model_trace.to_string(),
                "gt_trace": gt_trace.to_string() if gt_trace.is_valid else "GT also failed"
            }
        
        if not gt_trace.is_valid:
            return {
                "diagnosis": "GT program failed to execute (data issue)",
                "divergence_point": -1,
                "error_type": "GT_ERROR",
                "corrective_hint": "Ground truth program is invalid.",
                "model_trace": model_trace.to_string(),
                "gt_trace": "Invalid"
            }
        
        model_steps = model_trace.steps
        gt_steps = gt_trace.steps
        
        # Check structure match (number of steps)
        if len(model_steps) != len(gt_steps):
            error_type = "STRUCTURE"
            diagnosis_parts.append(
                f"STRUCTURE MISMATCH: Model has {len(model_steps)} steps, GT has {len(gt_steps)} steps."
            )
            corrective_hint = "The program structure is wrong. Review operation sequence."
            divergence_point = 0
        else:
            # Compare step by step
            for i, (m_step, gt_step) in enumerate(zip(model_steps, gt_steps)):
                # Check operation match
                if m_step.operation != gt_step.operation:
                    error_type = "OPERATION"
                    diagnosis_parts.append(
                        f"Step {i}: OPERATION MISMATCH - Model used '{m_step.operation}', GT used '{gt_step.operation}'"
                    )
                    corrective_hint = f"Wrong operation at step {i}. Expected {gt_step.operation}."
                    divergence_point = i
                    break
                
                # Check operand values
                if not self.compare_values(m_step.arg1_resolved, gt_step.arg1_resolved):
                    error_type = "OPERAND"
                    diagnosis_parts.append(
                        f"Step {i}: ARG1 MISMATCH - Model: {m_step.arg1_raw}={m_step.arg1_resolved}, "
                        f"GT: {gt_step.arg1_raw}={gt_step.arg1_resolved}"
                    )
                    corrective_hint = f"Wrong value extracted for first operand at step {i}."
                    divergence_point = i
                    break
                
                if not self.compare_values(m_step.arg2_resolved, gt_step.arg2_resolved):
                    error_type = "OPERAND"
                    diagnosis_parts.append(
                        f"Step {i}: ARG2 MISMATCH - Model: {m_step.arg2_raw}={m_step.arg2_resolved}, "
                        f"GT: {gt_step.arg2_raw}={gt_step.arg2_resolved}"
                    )
                    corrective_hint = f"Wrong value extracted for second operand at step {i}."
                    divergence_point = i
                    break
                
                # Check result
                if not self.compare_values(m_step.result, gt_step.result):
                    # This shouldn't happen if operands match
                    error_type = "ARITHMETIC"
                    diagnosis_parts.append(
                        f"Step {i}: RESULT MISMATCH - Model: {m_step.result}, GT: {gt_step.result}"
                    )
                    divergence_point = i
                    break
        
        if not diagnosis_parts:
            # Final results don't match but all steps seem okay
            if not self.compare_values(model_trace.final_result, gt_trace.final_result):
                error_type = "ROUNDING"
                diagnosis_parts.append(
                    f"FINAL RESULT MISMATCH: Model={model_trace.final_result}, GT={gt_trace.final_result}"
                )
        
        diagnosis = "\n".join(diagnosis_parts) if diagnosis_parts else "Traces match"
        
        return {
            "diagnosis": diagnosis,
            "divergence_point": divergence_point,
            "error_type": error_type,
            "corrective_hint": corrective_hint,
            "model_trace": model_trace.to_string(),
            "gt_trace": gt_trace.to_string()
        }


# Test the execution tracer
if __name__ == "__main__":
    print("Testing Execution Tracer...")
    print("=" * 60)
    
    # Test case 1: Basic program
    prog1 = ["subtract(", "100", "50", ")", "divide(", "#0", "50", ")", "EOF"]
    trace1 = eval_program_with_trace(prog1)
    print("Test 1: Growth rate calculation")
    print(trace1.to_string())
    print(f"Valid: {trace1.is_valid}")
    print()
    
    # Test case 2: Percentage input
    prog2 = ["divide(", "50%", "2", ")", "EOF"]
    trace2 = eval_program_with_trace(prog2)
    print("Test 2: Percentage handling")
    print(trace2.to_string())
    print()
    
    # Test case 3: Error case (division by zero)
    prog3 = ["divide(", "100", "0", ")", "EOF"]
    trace3 = eval_program_with_trace(prog3)
    print("Test 3: Division by zero")
    print(trace3.to_string())
    print()
    
    # Test case 4: Compare two traces
    print("Test 4: Trace comparison")
    prog_model = ["subtract(", "2012", "2011", ")", "divide(", "#0", "2011", ")", "EOF"]
    prog_gt = ["subtract(", "500", "400", ")", "divide(", "#0", "400", ")", "EOF"]
    
    trace_model = eval_program_with_trace(prog_model)
    trace_gt = eval_program_with_trace(prog_gt)
    
    comparator = TraceComparator()
    diff = comparator.diff_traces(trace_model, trace_gt)
    print(f"Diagnosis: {diff['diagnosis']}")
    print(f"Error type: {diff['error_type']}")
    print(f"Hint: {diff['corrective_hint']}")
    print()
    
    print("=" * 60)
    print("✅ Execution Tracer tests complete!")
