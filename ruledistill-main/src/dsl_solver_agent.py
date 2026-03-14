"""
DSL Solver Agent

This module implements a solver that generates FinQA Domain-Specific Language (DSL)
programs instead of direct numerical answers.

The DSL uses 6 arithmetic operations:
- add, subtract, multiply, divide, exp, greater

All numerical data is extracted from the provided context and used with arithmetic operations.

Example program: ["subtract(", "6348", "6241", ")", "divide(", "#0", "6241", ")", "EOF"]
"""

import re
import json
from typing import Optional, List, Dict
import xml.etree.ElementTree as ET

from model_client import LLMClient
from dsl_evaluator import eval_program, parse_program_from_string, compare_results
from prompt import DSL_SYSTEM_PROMPT, DSL_USER_PROMPT

# BM25 imports (optional - only needed if rule selection is enabled)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not installed. Rule selection will be disabled.")


class DSLSolverAgent:
    """
    Solver agent that generates FinQA DSL programs.
    
    Instead of directly answering questions, this agent generates executable
    programs that can be evaluated to produce numerical answers.
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.1, 
                 enable_rule_selection: bool = False, top_k_rules: int = 3,
                 backend: str = "ollama", ollama_model: str = "qwen3-next:latest",
                 think: bool = False):
        """
        Initialize the DSL solver.
        
        Args:
            model_name: LLM model name (defaults to config.MODEL_NAME for nvidia backend)
            temperature: Generation temperature
            enable_rule_selection: If True, use BM25 to select relevant rules
            top_k_rules: Number of top rules to select when rule_selection is enabled
            backend: LLM backend to use ("nvidia", "ollama", or None for config default)
            ollama_model: Model name for Ollama backend (e.g., "qwen3-next:latest")
            think: Enable thinking mode for Ollama models (e.g. qwen3-next)
        """
        # Unified LLM client
        self.llm = LLMClient(
            backend=backend,
            model_name=ollama_model if (backend or "ollama") == "ollama" else model_name,
            temperature=temperature,
            think=think,
        )
        self.temperature = temperature
        self.enable_rule_selection = enable_rule_selection
        self.top_k_rules = top_k_rules
        
        # BM25 components (initialized when rulebook is provided)
        self.bm25 = None
        self.rules = []
        self.corpus = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        # Remove non-alphanumeric and lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        return text.split()
    
    def _parse_rulebook_xml(self, rulebook: str) -> List[Dict]:
        """Parse XML rulebook into list of rule dictionaries."""
        rules = []
        try:
            root = ET.fromstring(rulebook)
            for rule_elem in root.findall('Rule'):
                trigger_elem = rule_elem.find('Trigger')
                action_elem = rule_elem.find('Action')
                example_elem = rule_elem.find('Example')
                
                rule_data = {
                    'id': rule_elem.get('id', ''),
                    'type': rule_elem.get('type', ''),
                    'trigger': trigger_elem.text.strip() if trigger_elem is not None and trigger_elem.text else '',
                    'action': action_elem.text.strip() if action_elem is not None and action_elem.text else '',
                    'example': example_elem.text.strip() if example_elem is not None and example_elem.text else ''
                }
                rules.append(rule_data)
        except ET.ParseError as e:
            print(f"Warning: Could not parse rulebook XML: {e}")
        
        return rules
    
    def _initialize_bm25(self, rulebook: str):
        """Initialize BM25 index from rulebook."""
        if not BM25_AVAILABLE:
            return
        
        # Parse rulebook
        self.rules = self._parse_rulebook_xml(rulebook)
        
        if not self.rules:
            return
        
        # Create corpus from triggers (the main matching text)
        self.corpus = [f"{rule['trigger']} {rule['type']}" for rule in self.rules]
        
        # Tokenize and initialize BM25
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"  BM25 initialized with {len(self.rules)} rules")
    
    def select_relevant_rules(self, question: str, rulebook: str) -> List[Dict]:
        """Select top-K relevant rules using BM25.
        
        Args:
            question: The question text to match against rule triggers
            rulebook: Full XML rulebook
            
        Returns:
            List of selected rule dictionaries
        """
        if not self.enable_rule_selection or not BM25_AVAILABLE:
            # Return all rules if selection is disabled
            return self._parse_rulebook_xml(rulebook)
        
        # Initialize BM25 if not already done or if rulebook changed
        if self.bm25 is None or not self.rules:
            self._initialize_bm25(rulebook)
        
        if not self.bm25 or not self.rules:
            return []
        
        # Search using question
        tokenized_query = self._tokenize(question)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k_rules]
        
        # Return selected rules with score > 0
        selected_rules = []
        for idx in top_indices:
            if scores[idx] > 0:
                selected_rules.append(self.rules[idx])
        
        return selected_rules
    
    def _format_rulebook(self, selected_rules: List[Dict], total_rules: int = None) -> str:
        """Format selected rules for inclusion in prompt.
        
        Args:
            selected_rules: List of rule dictionaries to include
            total_rules: Total number of rules in rulebook (for metadata)
            
        Returns:
            Formatted string for prompt
        """
        if not selected_rules:
            return ""
        
        # Build rules text
        rules_text = []
        for rule in selected_rules:
            rule_str = f"""**Rule {rule['id']}** (Type: {rule['type']})
Trigger: {rule['trigger']}
Action: {rule['action']}"""
            if rule.get('example'):
                rule_str += f"\nExample: {rule['example']}"
            rules_text.append(rule_str)
        
        metadata = ""
        if total_rules and self.enable_rule_selection:
            metadata = f" (selected {len(selected_rules)} from {total_rules} total rules)"
        
        return f"""
## Financial Reasoning Rules{metadata}

Apply these rules when generating your program:

{chr(10).join(rules_text)}
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
               table: List[List[str]] = None, timeout_s: float = None) -> dict:
        """
        Generate a DSL program for the given question.
        
        Args:
            question: The financial question
            context: The context information
            rulebook: Optional rulebook string
            table: Optional table data for table operations
            timeout_s: Optional wall-clock timeout in seconds. Uses streaming
                       internally when set. None = no timeout (blocking call).
            
        Returns:
            Dictionary with program, result, and metadata
        """
        # Select relevant rules if rule selection is enabled
        if rulebook and rulebook.strip():
            all_rules = self._parse_rulebook_xml(rulebook)
            selected_rules = self.select_relevant_rules(question, rulebook)
            rulebook_formatted = self._format_rulebook(selected_rules, len(all_rules))
        else:
            selected_rules = []
            rulebook_formatted = ""
        
        # Format prompts
        system_prompt = DSL_SYSTEM_PROMPT.format(
            rulebook=rulebook_formatted
        )
        user_prompt = DSL_USER_PROMPT.format(
            context=context,
            question=question
        )
        
        try:
            # Call LLM — use streaming + timeout when timeout_s is set
            if timeout_s is not None:
                llm_result = self.llm.chat_with_timeout(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout_s=timeout_s,
                )
            else:
                llm_result = self.llm.chat_with_metadata(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            raw_response = llm_result["content"]
            thinking = llm_result.get("thinking", "")
            timed_out = llm_result.get("timed_out", False)
            
            # Parse program
            program = self._parse_program_response(raw_response)
            
            if program is None:
                return {
                    "success": False,
                    "program": None,
                    "result": None,
                    "error": "Failed to parse program",
                    "raw_response": raw_response,
                    "thinking": thinking
                }
            
            # Execute program
            invalid_flag, result = eval_program(program, table)
            
            if invalid_flag:
                return {
                    "success": False,
                    "program": program,
                    "result": None,
                    "error": "Program execution failed",
                    "raw_response": raw_response,
                    "thinking": thinking
                }
            
            return {
                "success": True,
                "program": program,
                "result": result,
                "raw_response": raw_response,
                "thinking": thinking,
                "timed_out": timed_out
            }
            
        except Exception as e:
            return {
                "success": False,
                "program": None,
                "result": None,
                "error": str(e),
                "raw_response": None,
                "thinking": ""
            }
    
    def predict_batch(self, batch: List[dict], rulebook: str = "",
                      timeout_s: float = None) -> List[dict]:
        """
        Generate DSL programs for a batch of items.
        
        Args:
            batch: List of items with 'question', 'context', 'program' (gt), optionally 'table'
            rulebook: Optional rulebook string
            timeout_s: Optional per-question wall-clock timeout in seconds
            
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
                table=item.get("table"),
                timeout_s=timeout_s
            )
            
            # Add item metadata
            prediction["idx"] = item.get("idx", -1)
            prediction["question"] = item["question"]
            prediction["context"] = item.get("context", "")  # Needed by optimizer for failure analysis
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
