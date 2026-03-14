"""
Optimizer Agent

The Optimizer performs root cause analysis on Solver failures and evolves
the rulebook to prevent future errors. It implements the "self-regulating"
feedback loop.
"""

import re

import config
from model_client import LLMClient
from prompt import OPTIMIZER_SYSTEM_PROMPT, OPTIMIZER_USER_PROMPT, OPTIMIZER_XML_RETRY_PROMPT
from rulebook_utils import (
    parse_rulebook,
    serialize_rulebook,
    extract_rules_from_response,
    count_rules,
    compress_rulebook,
    get_empty_rulebook
)


class OptimizerAgent:
    """
    Analyzes batch failures and synthesizes rule updates.
    
    Input: Batch results (predictions, ground truths, reasoning) + Current Rulebook
    Output: Revised Rulebook XML
    """
    
    def __init__(self, client_type: str = "ollama", backend: str = None,
                 ollama_model: str = "qwen3-next:latest", think: bool = False):
        """
        Initialize the Optimizer agent.
        
        Args:
            client_type: Type of LLM client ("nvidia" for NVIDIA NIM) - deprecated, use backend
            backend: LLM backend to use ("nvidia", "ollama", or None for config default)
            ollama_model: Model name for Ollama backend (e.g., "qwen3-next:latest")
            think: Enable thinking mode for Ollama models (e.g. qwen3-next)
        """
        # Support both old client_type and new backend parameter
        effective_backend = backend or client_type or config.LLM_BACKEND
        
        # Unified LLM client
        self.llm = LLMClient(
            backend=effective_backend,
            model_name=ollama_model if effective_backend == "ollama" else None,
            temperature=0.3,
            max_tokens=None,
            think=think,
        )
    
    def optimize(
        self,
        batch_results: list[dict],
        current_rulebook: str,
        batch_num: int = 0,
        max_rules: int = 1000,
        timeout_s: float = None
    ) -> dict:
        """
        Analyze batch failures and generate optimized rulebook.
        
        Args:
            batch_results: List of prediction results with ground truths
            current_rulebook: Current rulebook XML string
            batch_num: Current batch number (for tracking)
            max_rules: Maximum number of rules allowed
            timeout_s: Optional wall-clock timeout in seconds per LLM call.
                       Uses streaming internally when set. None = no timeout.
            
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
        system_prompt = OPTIMIZER_SYSTEM_PROMPT
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
            print(f"[DEBUG] Calling LLM via model_client")
            
            if timeout_s is not None:
                llm_result = self.llm.chat_with_timeout(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout_s=timeout_s,
                )
                raw_response = llm_result["content"]
                if llm_result.get("timed_out"):
                    print(f"[Optimizer] ⚠ LLM call timed out after {timeout_s}s — using partial response")
            else:
                raw_response = self.llm.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            print(raw_response)
            
            # DEBUG: Log the raw response for troubleshooting
            print(f"[DEBUG] Optimizer raw response length: {len(raw_response)} chars")
            print(f"[DEBUG] First 200 chars: {raw_response[:200]}..." if len(raw_response) > 200 else f"[DEBUG] Full response: {raw_response}")
            
            # Extract and validate rulebook from response
            new_rulebook = extract_rules_from_response(raw_response)
            
            if not new_rulebook:
                # Try using the whole response as XML
                new_rulebook = raw_response
            
            # XML Validation with retry logic
            import xml.etree.ElementTree as ET
            xml_valid = False
            retry_count = 0
            max_retries = 1
            
            # Sanitize common XML issues from LLM output (e.g., S&P → S&amp;P)
            new_rulebook = self._sanitize_xml(new_rulebook)
            
            while not xml_valid and retry_count <= max_retries:
                try:
                    # Attempt to parse XML
                    ET.fromstring(new_rulebook)
                    xml_valid = True
                except ET.ParseError as e:
                    print(f"XML Parse Error: {e}")
                    
                    if retry_count < max_retries:
                        print(f"Retrying with stricter XML format prompt (attempt {retry_count + 1}/{max_retries})...")
                        
                        # Retry with stricter prompt
                        retry_system_prompt = system_prompt + OPTIMIZER_XML_RETRY_PROMPT
                        
                        if timeout_s is not None:
                            retry_result = self.llm.chat_with_timeout(
                                system_prompt=retry_system_prompt,
                                user_prompt=user_prompt,
                                timeout_s=timeout_s,
                            )
                            raw_response = retry_result["content"]
                            if retry_result.get("timed_out"):
                                print(f"[Optimizer] ⚠ Retry timed out after {timeout_s}s")
                        else:
                            raw_response = self.llm.chat(
                                system_prompt=retry_system_prompt,
                                user_prompt=user_prompt,
                                temperature=0.1,  # Lower temperature for stricter format
                            )
                        
                        # DEBUG: Log retry response
                        print(f"[DEBUG] Retry response length: {len(raw_response)} chars")
                        print(f"[DEBUG] Retry first 200 chars: {raw_response[:200]}..." if len(raw_response) > 200 else f"[DEBUG] Retry full: {raw_response}")
                        
                        new_rulebook = extract_rules_from_response(raw_response)
                        if not new_rulebook:
                            new_rulebook = raw_response
                        
                        retry_count += 1
                    else:
                        # Max retries reached, keep current rulebook
                        print("Warning: Could not parse optimizer response. Keeping current rulebook.")
                        new_rulebook = current_rulebook
                        xml_valid = True  # Exit loop
            
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
            
        except ConnectionError as e:
            print(f"❌ Connection error: {e}")
            print(f"   Make sure Ollama is running if using Ollama backend")
            return {
                "new_rulebook": current_rulebook,
                "analysis_summary": f"Connection error: {str(e)}",
                "metrics": analysis,
                "success": False
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
            # Use the solver's already-computed is_correct (from DSLEvaluator)
            # instead of re-comparing with a potentially mismatched key/tolerance
            is_correct = result.get("is_correct", False)
            
            if is_correct:
                correct += 1
            else:
                failures.append({
                    "idx": result.get("idx"),
                    "question": result.get("question", ""),
                    "context": result.get("context", ""),  # Supporting facts from gold_inds
                    "predicted": str(result.get("result", "N/A")),
                    "ground_truth": str(result.get("ground_truth", "")),
                    "program": result.get("program", ""),  # Generated DSL program
                    "gt_program": result.get("gt_program", ""),  # Ground truth program
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
            
            # Include context (supporting facts from gold_inds)
            context = f.get('context', '')
            if context:
                context_trunc = context[:300] + "..." if len(context) > 300 else context
                lines.append(f"**Context:** {context_trunc}")
            
            lines.append(f"**Predicted:** {f['predicted']}")
            lines.append(f"**Ground Truth:** {f['ground_truth']}")
            
            # Include generated and ground truth programs
            gen_program = f.get('program', '')
            if gen_program:
                lines.append(f"**Generated Program:** {str(gen_program)[:200]}..." if len(str(gen_program)) > 200 else f"**Generated Program:** {gen_program}")
            
            gt_program = f.get('gt_program', '')
            if gt_program:
                lines.append(f"**Ground Truth Program:** {str(gt_program)[:200]}..." if len(str(gt_program)) > 200 else f"**Ground Truth Program:** {gt_program}")
            
            lines.append(f"**Rules Applied:** {', '.join(f['rules_applied']) if f['rules_applied'] else 'none'}")
            
            # Include truncated reasoning
            reasoning = f['reasoning'][:500] + "..." if len(f['reasoning']) > 500 else f['reasoning']
            lines.append(f"**Reasoning:** {reasoning}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def _sanitize_xml(xml_str: str) -> str:
        """Sanitize LLM-generated XML by escaping bare & characters.
        
        LLMs commonly produce text like 'S&P 500' which is invalid XML.
        This escapes & characters that are NOT already part of valid XML
        entities (&amp; &lt; &gt; &quot; &apos; or &#...).
        """
        # Replace & that is NOT followed by amp; lt; gt; quot; apos; or #
        sanitized = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#)', '&amp;', xml_str)
        if sanitized != xml_str:
            print(f"[Optimizer] Sanitized {xml_str.count('&') - sanitized.count('&')} bare '&' in XML")
        return sanitized
    
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
