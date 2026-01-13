"""
Solver Agent

The Solver treats the Rulebook as "Law" and performs Chain-of-Thought (CoT)
reasoning based on the provided rules to answer financial questions.
"""

import re
import xml.etree.ElementTree as ET
from openai import OpenAI

import config


# Solver-specific prompts
SOLVER_SYSTEM_PROMPT = """You are a financial reasoning expert who STRICTLY follows the provided rulebook.

<Rulebook>
{rulebook}
</Rulebook>

## CRITICAL: CONTEXT TRUST
The Context provided is VERIFIED GROUND TRUTH extracted by an expert retrieval system.
- The Context is 100% CORRECT and COMPLETE for answering the question
- NEVER claim that information is "missing" or "not provided"
- The answer CAN and MUST be derived from the given Context
- If you think something is missing, re-read the Context - the answer IS there

Instructions:
1. Read and understand ALL rules in the rulebook before answering
2. TRUST the Context completely - it contains all information needed
3. Apply relevant rules during your step-by-step reasoning
4. Cite rule IDs when you use them (e.g., "Per Rule #03...")
5. If no rules exist yet, use your best financial reasoning judgment
6. Output your response in the specified XML format EXACTLY

CRITICAL: Your final answer must be a single numerical value. Do not include units, currency symbols, or explanations in the <Answer> tag."""

SOLVER_USER_PROMPT = """## TRUSTED CONTEXT (100% Verified - Use This Data Directly)
{context}

## Question
{question}

IMPORTANT: The Context above contains ALL the information needed to answer this question.
Do NOT claim any information is missing. Derive the answer from the Context provided.

Provide your answer in this EXACT XML format:
<Response>
    <Reasoning>Your step-by-step reasoning here, citing any rules you applied...</Reasoning>
    <Answer>Your numerical answer only (e.g., 0.25 or 1500000)</Answer>
    <RulesApplied>Comma-separated rule IDs you used, or "none" if no rules applied</RulesApplied>
</Response>"""


class SolverAgent:
    """
    Executes question-answering with rulebook-guided reasoning.
    
    Input: Question + Context + Current Rulebook
    Output: Structured response with reasoning trace and final answer
    """
    
    def __init__(self, client_type: str = "nvidia"):
        """
        Initialize the Solver agent.
        
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
    
    def predict(self, question: str, context: str, rulebook: str) -> dict:
        """
        Generate a prediction using Chain-of-Thought reasoning with the rulebook.
        
        Args:
            question: The financial question to answer
            context: Relevant context/data for the question
            rulebook: Current rulebook XML string
            
        Returns:
            Dictionary with keys:
                - reasoning: str (CoT reasoning trace)
                - answer: str (final numerical answer)
                - rules_applied: list[str] (rule IDs that were referenced)
                - raw_response: str (full LLM response)
                - success: bool (whether parsing succeeded)
        """
        # Format prompts
        system_prompt = SOLVER_SYSTEM_PROMPT.format(rulebook=rulebook)
        user_prompt = SOLVER_USER_PROMPT.format(context=context, question=question)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic for reproducibility
                max_tokens=1024,
                top_p=0.95
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Parse the XML response
            parsed = self._parse_response(raw_response)
            parsed["raw_response"] = raw_response
            
            return parsed
            
        except Exception as e:
            print(f"Solver error: {e}")
            return {
                "reasoning": "",
                "answer": "",
                "rules_applied": [],
                "raw_response": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_response(self, response: str) -> dict:
        """
        Parse the XML-formatted response from the Solver.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed dictionary with reasoning, answer, rules_applied, success
        """
        result = {
            "reasoning": "",
            "answer": "",
            "rules_applied": [],
            "success": False
        }
        
        try:
            # Try to extract XML from response
            xml_match = re.search(r'<Response>(.*?)</Response>', response, re.DOTALL | re.IGNORECASE)
            
            if xml_match:
                xml_content = f"<Response>{xml_match.group(1)}</Response>"
                root = ET.fromstring(xml_content)
                
                # Extract reasoning
                reasoning_elem = root.find("Reasoning")
                if reasoning_elem is not None and reasoning_elem.text:
                    result["reasoning"] = reasoning_elem.text.strip()
                
                # Extract answer
                answer_elem = root.find("Answer")
                if answer_elem is not None and answer_elem.text:
                    result["answer"] = self._clean_answer(answer_elem.text.strip())
                
                # Extract rules applied
                rules_elem = root.find("RulesApplied")
                if rules_elem is not None and rules_elem.text:
                    rules_text = rules_elem.text.strip().lower()
                    if rules_text != "none" and rules_text:
                        result["rules_applied"] = [r.strip() for r in rules_elem.text.split(",") if r.strip()]
                
                result["success"] = True
            else:
                # Fallback: try to extract answer from raw response
                result["answer"] = self._extract_answer_fallback(response)
                result["reasoning"] = response
                result["success"] = bool(result["answer"])
                
        except ET.ParseError:
            # XML parsing failed, try fallback
            result["answer"] = self._extract_answer_fallback(response)
            result["reasoning"] = response
            result["success"] = bool(result["answer"])
        except Exception as e:
            print(f"Parse error: {e}")
        
        return result
    
    def _clean_answer(self, answer: str) -> str:
        """
        Clean the answer string to extract just the numerical value.
        
        Args:
            answer: Raw answer string
            
        Returns:
            Cleaned numerical string
        """
        # Remove common prefixes/suffixes
        answer = answer.replace("$", "").replace(",", "").strip()
        answer = re.sub(r'^(approximately|about|roughly|around)\s*', '', answer, flags=re.IGNORECASE)
        
        # Handle percentage conversion
        if "%" in answer:
            match = re.search(r'(-?\d+\.?\d*)\s*%', answer)
            if match:
                return match.group(1)
        
        # Extract number
        match = re.search(r'-?\d+\.?\d*', answer)
        if match:
            return match.group(0)
        
        return answer
    
    def _extract_answer_fallback(self, response: str) -> str:
        """
        Fallback method to extract answer when XML parsing fails.
        
        Args:
            response: Raw response string
            
        Returns:
            Extracted answer or empty string
        """
        # Look for answer patterns
        patterns = [
            r'answer[:\s]+(-?\d+\.?\d*)',
            r'result[:\s]+(-?\d+\.?\d*)',
            r'=\s*(-?\d+\.?\d*)',
            r'(-?\d+\.?\d*)\s*$',  # Last number in response
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def predict_batch(self, batch: list[dict], rulebook: str) -> list[dict]:
        """
        Process a batch of questions.
        
        Args:
            batch: List of dictionaries with 'question' and 'context' keys
            rulebook: Current rulebook XML string
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        
        for item in batch:
            prediction = self.predict(
                question=item["question"],
                context=item["context"],
                rulebook=rulebook
            )
            
            # Include item metadata
            prediction["idx"] = item.get("idx")
            prediction["question"] = item["question"]
            prediction["ground_truth"] = item.get("ground_truth", "")
            
            results.append(prediction)
        
        return results
