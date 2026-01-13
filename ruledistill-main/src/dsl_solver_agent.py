"""
DSL Solver Agent

This module implements a solver that generates FinQA Domain-Specific Language (DSL)
programs instead of direct numerical answers.

The DSL uses operations like:
- add, subtract, multiply, divide, exp, greater
- table_max, table_min, table_sum, table_average

Example program: ["subtract(", "6348", "6241", ")", "divide(", "#0", "6241", ")", "EOF"]
"""

import re
import json
from typing import Optional, List
from openai import OpenAI

import config
from dsl_evaluator import eval_program, parse_program_from_string, compare_results


# DSL Solver System Prompt
DSL_SYSTEM_PROMPT = """You are a financial reasoning expert that generates executable programs using a Domain-Specific Language (DSL).

## Available Operations

You MUST use exactly these 10 operations:

### Arithmetic Operations (2 arguments)
| Operation | Description | Example |
|-----------|-------------|---------|
| add(a, b) | Addition | add(100, 200) → 300 |
| subtract(a, b) | Subtraction | subtract(500, 200) → 300 |
| multiply(a, b) | Multiplication | multiply(10, 5) → 50 |
| divide(a, b) | Division | divide(100, 4) → 25 |
| exp(a, b) | Exponentiation | exp(2, 3) → 8 |
| greater(a, b) | Comparison | greater(100, 50) → "yes" |

### Table Operations (row_name, "none")
| Operation | Description | Example |
|-----------|-------------|---------|
| table_max(row, none) | Maximum value in row | table_max(revenue, none) |
| table_min(row, none) | Minimum value in row | table_min(costs, none) |
| table_sum(row, none) | Sum of row values | table_sum(total, none) |
| table_average(row, none) | Average of row values | table_average(income, none) |

## Argument Types

1. **Numbers**: Use exact values from context (e.g., "6348", "0.05", "1500000")
2. **Percentages**: Write as "50%" (automatically converts to 0.5)
3. **Constants**: Use "const_100" for 100, "const_m1" for -1
4. **Step References**: Use "#0", "#1" to reference previous step results

## Output Format

Output your program as a JSON array of tokens:
```json
["operation(", "arg1", "arg2", ")", ..., "EOF"]
```

## Rules

1. Every program MUST end with "EOF"
2. Each operation is 4 tokens: ["op(", "arg1", "arg2", ")"]
3. Use "#N" to reference step N's result (0-indexed)
4. Extract exact numbers from the context - don't round or modify them
5. For percentage calculations, use the decimal form OR use divide by const_100

## Examples

### Example 1: Percentage Change
Question: "What is the growth rate from 6241 to 6348?"
Program:
```json
["subtract(", "6348", "6241", ")", "divide(", "#0", "6241", ")", "EOF"]
```
Explanation: (6348-6241)/6241 = 0.01715 or 1.715%

### Example 2: Percentage of Total
Question: "What percentage is 1733 of 2640?"
Program:
```json
["divide(", "1733", "2640", ")", "EOF"]
```
Explanation: 1733/2640 = 0.6564 or 65.64%

### Example 3: Year-over-Year Change
Question: "What is the difference between 2019 revenue (500) and 2018 revenue (450)?"
Program:
```json
["subtract(", "500", "450", ")", "EOF"]
```

### Example 4: Multi-step Calculation
Question: "If revenue is 1000 and costs are 600, what is the profit margin?"
Program:
```json
["subtract(", "1000", "600", ")", "divide(", "#0", "1000", ")", "EOF"]
```
Explanation: (1000-600)/1000 = 0.4 or 40%

{rulebook}

## CRITICAL FORMAT RULES (Read Carefully!)

### 1. PERCENTAGE RESULTS → Return as DECIMAL
When asked for percentage/portion/share, return DECIMAL (0.65), NOT multiplied by 100 (65).
```
✓ CORRECT: ["divide(", "1733", "2640", ")", "EOF"] → 0.6564
✗ WRONG:   [..., "multiply(", "#0", "100", ")", "EOF"] → 65.64
```

### 2. UNIT CONSISTENCY → Numbers are already in stated units
If context says "in millions" and numbers are 24490, 15386 - they're ALREADY in millions.
```
✓ CORRECT: ["subtract(", "24490", "15386", ")", "EOF"] → 9104 (millions)
✗ WRONG:   [..., "divide(", "#0", "1000", ")", "EOF"] → 9.1 (wrong scale)
```

### 3. EXACT NUMBERS → No rounding
Use exact values: 959.2 not 959, 23.6% not 24%.

IMPORTANT: Generate ONLY the JSON array. No explanations or additional text."""


DSL_USER_PROMPT = """## TRUSTED CONTEXT (100% Verified - Use This Data Directly)
{context}

## Question
{question}

Based on the context above, generate a DSL program to compute the answer.

Output ONLY the JSON array of tokens. Example format:
["subtract(", "6348", "6241", ")", "divide(", "#0", "6241", ")", "EOF"]"""


class DSLSolverAgent:
    """
    Solver agent that generates FinQA DSL programs.
    
    Instead of directly answering questions, this agent generates executable
    programs that can be evaluated to produce numerical answers.
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.1):
        """
        Initialize the DSL solver.
        
        Args:
            model_name: LLM model name (defaults to config.MODEL_NAME)
            temperature: Generation temperature
        """
        self.model_name = model_name or config.MODEL_NAME
        self.temperature = temperature
        self.client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY
        )
    
    def _format_rulebook(self, rulebook: str) -> str:
        """Format rulebook for inclusion in prompt."""
        if not rulebook or rulebook.strip() == "":
            return ""
        
        return f"""
## Financial Reasoning Rules

Apply these rules when generating your program:

<Rules>
{rulebook}
</Rules>
"""
    
    def _parse_program_response(self, response: str) -> Optional[List[str]]:
        """
        Parse the LLM response to extract the DSL program.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of program tokens or None if parsing fails
        """
        response = response.strip()
        
        # Try to find JSON array in response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                program = json.loads(json_match.group())
                if isinstance(program, list) and len(program) > 0:
                    # Ensure EOF is present
                    if program[-1] != "EOF":
                        program.append("EOF")
                    return program
            except json.JSONDecodeError:
                pass
        
        # Try to parse as function calls
        program = parse_program_from_string(response)
        if program:
            return program
        
        return None
    
    def predict(self, question: str, context: str, rulebook: str = "",
               table: List[List[str]] = None) -> dict:
        """
        Generate a DSL program for the given question.
        
        Args:
            question: The financial question
            context: The context information
            rulebook: Optional rulebook string
            table: Optional table data for table operations
            
        Returns:
            Dictionary with program, result, and metadata
        """
        # Format prompts
        system_prompt = DSL_SYSTEM_PROMPT.format(
            rulebook=self._format_rulebook(rulebook)
        )
        user_prompt = DSL_USER_PROMPT.format(
            context=context,
            question=question
        )
        
        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=512,
                top_p=0.95
            )
            
            raw_response = response.choices[0].message.content
            
            # Parse program
            program = self._parse_program_response(raw_response)
            
            if program is None:
                return {
                    "success": False,
                    "program": None,
                    "result": None,
                    "error": "Failed to parse program",
                    "raw_response": raw_response
                }
            
            # Execute program
            invalid_flag, result = eval_program(program, table)
            
            if invalid_flag:
                return {
                    "success": False,
                    "program": program,
                    "result": None,
                    "error": "Program execution failed",
                    "raw_response": raw_response
                }
            
            return {
                "success": True,
                "program": program,
                "result": result,
                "raw_response": raw_response
            }
            
        except Exception as e:
            return {
                "success": False,
                "program": None,
                "result": None,
                "error": str(e),
                "raw_response": None
            }
    
    def predict_batch(self, batch: List[dict], rulebook: str = "") -> List[dict]:
        """
        Generate DSL programs for a batch of items.
        
        Args:
            batch: List of items with 'question', 'context', 'program' (gt), optionally 'table'
            rulebook: Optional rulebook string
            
        Returns:
            List of prediction results with program comparison
        """
        from dsl_evaluator import DSLEvaluator, parse_program_from_string
        
        evaluator = DSLEvaluator(tolerance=0.01)
        results = []
        
        for item in batch:
            prediction = self.predict(
                question=item["question"],
                context=item["context"],
                rulebook=rulebook,
                table=item.get("table")
            )
            
            # Add item metadata
            prediction["idx"] = item.get("idx", -1)
            prediction["question"] = item["question"]
            prediction["ground_truth"] = item.get("ground_truth", "")
            prediction["gt_program"] = item.get("program", "")  # Ground truth program string
            
            # Evaluate by comparing with GT program execution
            gt_program = item.get("program", "")
            if prediction["success"] and prediction["program"] and gt_program:
                # Execute GT program and compare results
                eval_result = evaluator.evaluate_with_gt_program(
                    model_program=prediction["program"],
                    gt_program=gt_program,
                    table=item.get("table")
                )
                
                prediction["is_correct"] = eval_result.get("is_correct", False)
                prediction["program_match"] = eval_result.get("program_match", False)
                prediction["gt_result"] = eval_result.get("gt_result")
                prediction["gt_program_tokens"] = eval_result.get("gt_program")
            else:
                prediction["is_correct"] = False
                prediction["program_match"] = False
                prediction["gt_result"] = None
                prediction["gt_program_tokens"] = None
            
            results.append(prediction)
        
        return results


# Test the DSL solver
if __name__ == "__main__":
    print("Testing DSL Solver Agent...")
    print("=" * 60)
    
    # Create solver
    solver = DSLSolverAgent()
    
    # Test case
    test_item = {
        "question": "what percentage of total reorganization items net consisted of labor-related deemed claim?",
        "context": "labor-related deemed claim of 2013 is $1733. total reorganization items net of 2013 is $2640.",
        "ground_truth": 0.65644
    }
    
    print(f"Question: {test_item['question']}")
    print(f"Context: {test_item['context']}")
    print(f"Expected: {test_item['ground_truth']}")
    print()
    
    result = solver.predict(
        question=test_item["question"],
        context=test_item["context"]
    )
    
    print(f"Program: {result.get('program')}")
    print(f"Result: {result.get('result')}")
    print(f"Success: {result.get('success')}")
    
    if result.get('success') and result.get('result') is not None:
        is_correct = compare_results(result['result'], test_item['ground_truth'])
        print(f"Correct: {'✅ Yes' if is_correct else '❌ No'}")
