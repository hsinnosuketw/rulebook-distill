"""
Judger — evaluates LLM responses against ground truth.

Uses the shared LLMClient for backend-agnostic chat completions.
"""

from model_client import LLMClient
import prompt


class Judger:
    def __init__(self, backend: str = None):
        """
        Initialize the Judger.

        Args:
            backend: LLM backend ("nvidia", "ollama", or None for config default)
        """
        self.llm = LLMClient(backend=backend)

    def evaluate(self, question, ground_truth, response):
        """Evaluate a response against ground truth using an LLM judge."""
        formatted_user_prompt = prompt.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            response=response,
        )

        result = self.llm.chat(
            system_prompt=prompt.JUDGER_SYSTEM_PROMPT,
            user_prompt=formatted_user_prompt,
        )
        return result
