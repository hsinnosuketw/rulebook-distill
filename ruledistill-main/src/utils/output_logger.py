"""
Output Logger

Stores per-sample model outputs (question, answer, context, model response,
thinking, executed answer) to a JSONL file for offline analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class OutputLogger:
    """
    Logs model prediction details to a JSONL file.

    Usage::

        logger = OutputLogger("data/checkpoints/test-v2/model_outputs.jsonl")

        # After each prediction
        logger.log(
            question="What is revenue growth?",
            answer="20%",            # ground truth answer
            exe_ans=0.2,             # executed answer from DSL program
            context="Revenue was 100M in 2020, 120M in 2021.",
            model_response='["subtract(", "120", "100", ")", ...]',
            model_thinking="I need to calculate (120-100)/100...",
        )

        # Or log an entire batch
        logger.log_batch(solver_results, batch_items)
    """

    def __init__(self, output_path: str):
        """
        Args:
            output_path: Path to the output JSONL file.
                         Parent directories are created automatically.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-sample logging
    # ------------------------------------------------------------------

    def log(
        self,
        question: str,
        answer: str,
        exe_ans: Union[str, float, None],
        context: str,
        model_response: str,
        model_thinking: str = "",
        *,
        # Optional extras
        idx: int = None,
        program: list = None,
        gt_program: str = None,
        is_correct: bool = None,
        error: str = None,
        batch_num: int = None,
        epoch: int = None,
    ):
        """
        Write a single record to the log file.

        Args:
            question: The input question
            answer: Ground-truth answer string
            exe_ans: Executed answer from the DSL program
            context: Gold context provided to the model
            model_response: Raw model output (content field)
            model_thinking: Model's chain-of-thought / thinking field
            idx: Sample index
            program: Parsed DSL program tokens
            gt_program: Ground-truth program string
            is_correct: Whether the answer matched
            error: Error message if execution failed
            batch_num: Batch number
            epoch: Epoch number
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "idx": idx,
            "question": question,
            "answer": answer,
            "exe_ans": _serialize(exe_ans),
            "context": context,
            "model_response": model_response,
            "model_thinking": model_thinking,
            # Extras
            "program": program,
            "gt_program": gt_program,
            "is_correct": is_correct,
            "error": error,
            "batch_num": batch_num,
            "epoch": epoch,
        }

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Batch logging  (convenience wrapper)
    # ------------------------------------------------------------------

    def log_batch(
        self,
        solver_results: list,
        batch_items: list,
        batch_num: int = None,
        epoch: int = None,
    ):
        """
        Log an entire batch of solver results.

        Args:
            solver_results: List of dicts returned by DSLSolverAgent.predict_batch
            batch_items: Corresponding list of data items from the DataBuffer
            batch_num: Current batch number
            epoch: Current epoch number
        """
        for result, item in zip(solver_results, batch_items):
            self.log(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                exe_ans=result.get("result"),
                context=item.get("context", ""),
                model_response=result.get("raw_response", ""),
                model_thinking=result.get("thinking", ""),
                idx=result.get("idx"),
                program=result.get("program"),
                gt_program=item.get("program", ""),
                is_correct=result.get("is_correct"),
                error=result.get("error"),
                batch_num=batch_num,
                epoch=epoch,
            )

    # ------------------------------------------------------------------
    # Reading logged outputs
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: str) -> list:
        """Load all records from a JSONL log file."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def _serialize(value):
    """Convert non-JSON-serializable values to strings."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)

