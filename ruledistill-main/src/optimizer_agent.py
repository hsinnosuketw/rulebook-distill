"""
Optimizer Agent

The Optimizer performs root cause analysis on Solver failures and evolves
the rulebook to prevent future errors. It implements the "self-regulating"
feedback loop.
"""

import re
from openai import OpenAI

import config
from rulebook_utils import (
    parse_rulebook,
    serialize_rulebook,
    extract_rules_from_response,
    count_rules,
    compress_rulebook,
    get_empty_rulebook
)


# Optimizer-specific prompts
OPTIMIZER_SYSTEM_PROMPT = """You are an AI Alignment Engineer optimizing a rulebook for financial QA.

Your task is to analyze solver failures and evolve the rulebook to prevent future errors.

## Guidelines

### GENERALIZATION (Critical)
- Rules must be ABSTRACT and applicable to unseen cases
- Focus on methodology and logic patterns, NOT specific values

### NO DATA LEAKAGE (Critical)
- NEVER include specific numbers, entity names, or values from test cases in rules
- Bad: "When calculating Apple's revenue, add Q1 + Q2"
- Good: "When calculating annual revenue, sum all quarterly values"

### STABILITY
- Only modify rules that fail multiple times in the batch (2+ failures)
- Do not oscillate on rule wording

### COMPRESSION
- Keep total rules under {max_rules}
- Merge similar rules when possible
- Remove rules that consistently don't help

### ERROR TYPES
Classify each failure as one of:
- MISSING_RULE: No existing rule covers this case → Create new rule
- BAD_RULE: Existing rule led to error → Refine rule wording  
- CONFLICTING_RULES: Multiple rules contradicted → Merge or delete
- HALLUCINATION: Model ignored rules → Add stricter enforcement language
- ARITHMETIC_ERROR: Calculation mistake → Add verification instruction

## Output Format
Output ONLY the complete revised rulebook in XML format. No explanations before or after.
"""

OPTIMIZER_USER_PROMPT = """## Current Rulebook
{current_rulebook}

## Batch Results Summary
- Total questions: {total_count}
- Correct: {correct_count} ({accuracy:.1%})
- Failed: {error_count}

## Failure Analysis
{failure_analysis}

## Task
1. Analyze each failure's root cause
2. Classify error types
3. Generate revised rulebook (max {max_rules} rules)

Output the COMPLETE revised rulebook:
<Rulebook domain="finqa">
    <Rule id="01" type="..." phase="generation" confidence="1" source="batch_{batch_num}">
        <Trigger>...</Trigger>
        <Action>...</Action>
    </Rule>
    ...
</Rulebook>"""


class OptimizerAgent:
    """
    Analyzes batch failures and synthesizes rule updates.
    
    Input: Batch results (predictions, ground truths, reasoning) + Current Rulebook
    Output: Revised Rulebook XML
    """
    
    def __init__(self, client_type: str = "nvidia"):
        """
        Initialize the Optimizer agent.
        
        Args:
            client_type: Type of LLM client ("nvidia" for NVIDIA NIM)
        """
        if client_type == "nvidia":
            self.client = OpenAI(
                base_url=config.NVIDIA_BASE_URL,
                api_key=config.NVIDIA_API_KEY
            )
        else:
            raise ValueError(f"Unsupported client type: {client_type}")
        
        self.model_name = config.MODEL_NAME
    
    def optimize(
        self,
        batch_results: list[dict],
        current_rulebook: str,
        batch_num: int = 0,
        max_rules: int = 15
    ) -> dict:
        """
        Analyze batch failures and generate optimized rulebook.
        
        Args:
            batch_results: List of prediction results with ground truths
            current_rulebook: Current rulebook XML string
            batch_num: Current batch number (for tracking)
            max_rules: Maximum number of rules allowed
            
        Returns:
            Dictionary with:
                - new_rulebook: str (revised rulebook XML)
                - analysis_summary: str (explanation of changes)
                - metrics: dict (accuracy, error counts, etc.)
                - success: bool
        """
        # Analyze batch results
        analysis = self._analyze_batch(batch_results)
        
        # If no failures, return current rulebook unchanged
        if analysis["error_count"] == 0:
            return {
                "new_rulebook": current_rulebook,
                "analysis_summary": "All predictions correct. No rulebook changes needed.",
                "metrics": analysis,
                "success": True
            }
        
        # Build failure analysis string
        failure_analysis = self._format_failure_analysis(analysis["failures"])
        
        # Format prompts
        system_prompt = OPTIMIZER_SYSTEM_PROMPT.format(max_rules=max_rules)
        user_prompt = OPTIMIZER_USER_PROMPT.format(
            current_rulebook=current_rulebook,
            total_count=analysis["total_count"],
            correct_count=analysis["correct_count"],
            error_count=analysis["error_count"],
            accuracy=analysis["accuracy"],
            failure_analysis=failure_analysis,
            max_rules=max_rules,
            batch_num=batch_num
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Slight creativity for rule synthesis
                max_tokens=2048,
                top_p=0.95
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Extract and validate rulebook from response
            new_rulebook = extract_rules_from_response(raw_response)
            
            if not new_rulebook:
                # Try using the whole response as XML
                new_rulebook = raw_response
            
            # Validate and compress if needed
            rule_count = count_rules(new_rulebook)
            
            if rule_count > max_rules:
                new_rulebook = compress_rulebook(new_rulebook, max_rules)
            
            if rule_count == 0:
                # Parsing failed, keep current rulebook
                print("Warning: Could not parse optimizer response. Keeping current rulebook.")
                new_rulebook = current_rulebook
            
            return {
                "new_rulebook": new_rulebook,
                "analysis_summary": raw_response,
                "metrics": analysis,
                "success": True
            }
            
        except Exception as e:
            print(f"Optimizer error: {e}")
            return {
                "new_rulebook": current_rulebook,
                "analysis_summary": f"Error: {str(e)}",
                "metrics": analysis,
                "success": False
            }
    
    def _analyze_batch(self, batch_results: list[dict]) -> dict:
        """
        Analyze batch results to identify correct/incorrect predictions.
        
        Args:
            batch_results: List of prediction result dictionaries
            
        Returns:
            Analysis dictionary with counts and failure details
        """
        total = len(batch_results)
        correct = 0
        failures = []
        
        for result in batch_results:
            predicted = result.get("answer", "")
            ground_truth = result.get("ground_truth", "")
            
            is_correct = self._compare_answers(predicted, ground_truth)
            
            if is_correct:
                correct += 1
            else:
                failures.append({
                    "idx": result.get("idx"),
                    "question": result.get("question", ""),
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                    "reasoning": result.get("reasoning", ""),
                    "rules_applied": result.get("rules_applied", [])
                })
        
        return {
            "total_count": total,
            "correct_count": correct,
            "error_count": len(failures),
            "accuracy": correct / total if total > 0 else 0,
            "failures": failures
        }
    
    def _compare_answers(self, predicted: str, ground_truth, use_strict: bool = True) -> bool:
        """
        Compare predicted answer with ground truth.
        
        Uses the OFFICIAL FinQA evaluation approach:
        - Exact match (exe_res == gold_res) as the baseline
        - Minimal tolerance (1e-9) only for floating-point arithmetic errors
        - NO percentage↔decimal conversion (this was hiding errors)
        - NO scale factor matching (this was semantically wrong)
        
        Args:
            predicted: Predicted answer string
            ground_truth: Ground truth (can be float or string)
            use_strict: If True, use exact match like official FinQA.
                       If False, use 1% relative tolerance for lenient mode.
            
        Returns:
            True if answers match (exact or within floating-point epsilon)
        """
        try:
            # Parse predicted value (FinQA str_to_num style)
            pred_str = str(predicted).replace(",", "").replace("$", "").strip()
            if "%" in pred_str:
                pred_str = pred_str.replace("%", "")
                pred_val = float(pred_str) / 100.0  # Convert percentage to decimal
            elif "const" in pred_str:
                pred_str = pred_str.replace("const_", "")
                if pred_str == "m1":
                    pred_str = "-1"
                pred_val = float(pred_str)
            else:
                pred_val = float(pred_str)
            
            # Parse ground truth value (same FinQA str_to_num style)
            gt_str = str(ground_truth).replace(",", "").replace("$", "").strip()
            if "%" in gt_str:
                gt_str = gt_str.replace("%", "")
                gt_val = float(gt_str) / 100.0
            elif "const" in gt_str:
                gt_str = gt_str.replace("const_", "")
                if gt_str == "m1":
                    gt_str = "-1"
                gt_val = float(gt_str)
            else:
                gt_val = float(gt_str)
            
            if use_strict:
                # OFFICIAL FinQA: Exact match with floating-point epsilon
                # This mirrors: if exe_res == gold_res: exe_correct += 1
                # We use 1e-9 to handle Python floating-point arithmetic errors only
                if gt_val == 0:
                    return abs(pred_val) < 1e-9
                return abs(pred_val - gt_val) < 1e-9 or pred_val == gt_val
            else:
                # Lenient mode: 1% relative tolerance (still stricter than before)
                if abs(gt_val) < 1e-9:
                    return abs(pred_val) < 1e-6
                rel_diff = abs(pred_val - gt_val) / abs(gt_val)
                return rel_diff < 0.01  # 1% relative tolerance
            
        except (ValueError, TypeError):
            # Fallback to string comparison for non-numeric answers (yes/no)
            return str(predicted).strip().lower() == str(ground_truth).strip().lower()
    
    def _format_failure_analysis(self, failures: list[dict]) -> str:
        """
        Format failure details for the optimizer prompt.
        
        Args:
            failures: List of failure dictionaries
            
        Returns:
            Formatted string for the prompt
        """
        if not failures:
            return "No failures to analyze."
        
        lines = []
        for i, f in enumerate(failures, 1):
            lines.append(f"### Failure {i}")
            lines.append(f"**Question:** {f['question'][:200]}...")
            lines.append(f"**Predicted:** {f['predicted']}")
            lines.append(f"**Ground Truth:** {f['ground_truth']}")
            lines.append(f"**Rules Applied:** {', '.join(f['rules_applied']) if f['rules_applied'] else 'none'}")
            
            # Include truncated reasoning
            reasoning = f['reasoning'][:500] + "..." if len(f['reasoning']) > 500 else f['reasoning']
            lines.append(f"**Reasoning:** {reasoning}")
            lines.append("")
        
        return "\n".join(lines)
    
    def classify_error(self, failure: dict, current_rules: list[dict]) -> str:
        """
        Classify the type of error for a single failure.
        
        Args:
            failure: Failure dictionary
            current_rules: List of current rule dictionaries
            
        Returns:
            Error type string
        """
        rules_applied = failure.get("rules_applied", [])
        reasoning = failure.get("reasoning", "")
        
        if not rules_applied or rules_applied == ["none"]:
            # No rules were applied
            if not current_rules:
                return "MISSING_RULE"
            else:
                return "HALLUCINATION"
        
        # Rules were applied but still wrong
        # Check if multiple rules might have conflicted
        if len(rules_applied) > 1:
            return "CONFLICTING_RULES"
        
        # Single rule applied but wrong → bad rule
        return "BAD_RULE"
