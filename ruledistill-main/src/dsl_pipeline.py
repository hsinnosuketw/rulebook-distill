"""
DSL-Based Self-Regulated Pipeline

This pipeline uses the DSL Solver Agent to generate FinQA programs instead of 
direct numerical answers. The programs are then executed and evaluated.

This is an EXPERIMENTAL alternative to the regular pipeline.

Usage:
    python dsl_pipeline.py --dataset /path/to/train.json --batch-size 10

Key differences from regular pipeline:
1. Solver generates DSL programs (e.g., ["subtract(", "100", "50", ")", "EOF"])
2. Programs are executed using FinQA's eval_program()
3. Results are compared against exe_ans ground truth
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_buffer import DataBuffer
from dsl_solver_agent import DSLSolverAgent
from dsl_evaluator import DSLEvaluator, compare_results
from optimizer_agent import OptimizerAgent

import config


class DSLPipeline:
    """
    Self-regulated pipeline using DSL program generation.
    
    Flow:
    1. DSL Solver generates programs for each question
    2. Programs are executed to get numerical results
    3. Results are compared against ground truth
    4. Optimizer analyzes failures and updates rulebook
    """
    
    def __init__(
        self,
        solver: DSLSolverAgent,
        optimizer: OptimizerAgent,
        evaluator: DSLEvaluator,
        checkpoint_dir: str = "data/checkpoints/dsl",
        seed_rulebook_path: str = None
    ):
        """
        Initialize the DSL pipeline.
        
        Args:
            solver: DSL solver agent
            optimizer: Optimizer agent for rulebook updates
            evaluator: DSL program evaluator
            checkpoint_dir: Directory for saving checkpoints
            seed_rulebook_path: Optional path to seed rulebook XML
        """
        self.solver = solver
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load seed rulebook if provided
        self.rulebook = ""
        if seed_rulebook_path:
            try:
                with open(seed_rulebook_path, 'r') as f:
                    self.rulebook = f.read()
                print(f"Loaded seed rulebook from {seed_rulebook_path} ({len(self.rulebook)} chars)")
            except Exception as e:
                print(f"Warning: Could not load seed rulebook: {e}")
        
        # Metrics tracking
        self.metrics_log = []
    
    def _save_results(self, results: list, batch_num: int):
        """Save batch results to JSONL file."""
        results_file = self.checkpoint_dir / f"dsl_results_batch_{batch_num:03d}.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                # Make result JSON serializable
                serializable = {
                    "idx": result.get("idx"),
                    "question": result.get("question"),
                    "model_program": result.get("program"),
                    "model_result": result.get("result") if result.get("result") != "n/a" else None,
                    "gt_program": result.get("gt_program"),
                    "gt_result": result.get("gt_result"),
                    "is_correct": result.get("is_correct"),
                    "program_match": result.get("program_match"),
                    "success": result.get("success"),
                    "error": result.get("error")
                }
                f.write(json.dumps(serializable) + "\n")
    
    def _save_rulebook(self, batch_num: int):
        """Save current rulebook."""
        rulebook_file = self.checkpoint_dir / f"dsl_rulebook_batch_{batch_num:03d}.xml"
        with open(rulebook_file, "w") as f:
            f.write(self.rulebook)
    
    def _save_metrics(self, metrics: dict):
        """Append metrics to log file."""
        metrics_file = self.checkpoint_dir / "dsl_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def _format_failures_for_optimizer(self, results: list) -> list:
        """Format failed results for the optimizer in the expected format."""
        failures = []
        for result in results:
            if not result.get("is_correct", False):
                # Format model program as readable string
                model_prog = result.get("program", [])
                if isinstance(model_prog, list):
                    model_prog_str = " ".join(str(t) for t in model_prog if t != "EOF")
                else:
                    model_prog_str = str(model_prog)
                
                failures.append({
                    "idx": result.get("idx"),
                    "question": str(result.get("question", "")),
                    "predicted": str(result.get("result")) if result.get("result") is not None else "EXECUTION_ERROR",
                    "ground_truth": str(result.get("gt_result", "")),
                    "reasoning": f"Model program: {model_prog_str}",
                    "rules_applied": [],
                    "error": result.get("error", "Result mismatch"),
                    "gt_program": str(result.get("gt_program", ""))
                })
        return failures
    
    def run_batch(self, batch: list, batch_num: int) -> dict:
        """
        Run a single batch through the pipeline.
        
        Args:
            batch: List of items with question, context, ground_truth
            batch_num: Batch number for logging
            
        Returns:
            Dictionary with batch metrics
        """
        # Stage 1: Generate DSL programs
        print("Stage 1: DSL Solver generating programs...")
        solver_results = self.solver.predict_batch(batch, self.rulebook)
        
        # Calculate metrics
        total = len(solver_results)
        correct = sum(1 for r in solver_results if r.get("is_correct", False))
        program_matches = sum(1 for r in solver_results if r.get("program_match", False))
        execution_errors = sum(1 for r in solver_results if not r.get("success", False))
        accuracy = correct / total if total > 0 else 0
        program_match_rate = program_matches / total if total > 0 else 0
        
        # Save results
        self._save_results(solver_results, batch_num)
        
        # Stage 2: Analyze failures and update rulebook
        failures = self._format_failures_for_optimizer(solver_results)
        
        if failures and batch_num > 0:  # Skip optimizer on first batch
            print("Stage 2: Optimizer analyzing failures...")
            try:
                # optimizer.optimize returns a dict with 'new_rulebook' key
                optimizer_result = self.optimizer.optimize(
                    batch_results=solver_results,  # Pass full results, not just failures
                    current_rulebook=self.rulebook,
                    batch_num=batch_num
                )
                if optimizer_result.get("success") and optimizer_result.get("new_rulebook"):
                    new_rulebook = optimizer_result["new_rulebook"]
                    if new_rulebook and new_rulebook.strip():
                        self.rulebook = new_rulebook
                        self._save_rulebook(batch_num)
            except Exception as e:
                print(f"Warning: Optimizer failed: {e}")
        
        # Log metrics
        metrics = {
            "batch_num": batch_num,
            "total_count": total,
            "correct_count": correct,
            "program_match_count": program_matches,
            "execution_errors": execution_errors,
            "accuracy": accuracy,
            "program_match_rate": program_match_rate,
            "rulebook_size": len(self.rulebook),
            "timestamp": datetime.now().isoformat()
        }
        self._save_metrics(metrics)
        self.metrics_log.append(metrics)
        
        return metrics
    
    def run(
        self,
        data_buffer: DataBuffer,
        max_batches: Optional[int] = None,
        start_batch: int = 0
    ) -> dict:
        """
        Run the full pipeline.
        
        Args:
            data_buffer: Data buffer containing the dataset
            max_batches: Maximum number of batches to process
            start_batch: Starting batch number
            
        Returns:
            Summary statistics
        """
        print(f"\n{'='*60}")
        print("DSL-BASED SELF-REGULATED PIPELINE")
        print(f"{'='*60}")
        print(f"Total batches: {len(data_buffer)}")
        print(f"Batch size: {data_buffer.batch_size}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        total_correct = 0
        total_count = 0
        total_execution_errors = 0
        
        for batch_num, batch in enumerate(data_buffer):
            if batch_num < start_batch:
                continue
            
            if max_batches and batch_num >= start_batch + max_batches:
                break
            
            print(f"\n--- Batch {batch_num}/{len(data_buffer)} ---")
            
            metrics = self.run_batch(batch, batch_num)
            
            total_count += metrics["total_count"]
            total_correct += metrics["correct_count"]
            total_execution_errors += metrics["execution_errors"]
            
            print(f"Exec Accuracy: {metrics['accuracy']*100:.1f}% ({metrics['correct_count']}/{metrics['total_count']})")
            print(f"Program Match: {metrics['program_match_rate']*100:.1f}% ({metrics['program_match_count']}/{metrics['total_count']})")
            print(f"Execution errors: {metrics['execution_errors']}")
            print(f"Rulebook size: {metrics['rulebook_size']} chars")
        
        # Final summary
        overall_accuracy = total_correct / total_count if total_count > 0 else 0
        
        summary = {
            "total_count": total_count,
            "total_correct": total_correct,
            "execution_errors": total_execution_errors,
            "overall_accuracy": overall_accuracy,
            "batches_processed": len(self.metrics_log)
        }
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {total_count}")
        print(f"Correct: {total_correct} ({overall_accuracy*100:.2f}%)")
        print(f"Execution errors: {total_execution_errors}")
        print(f"Batches processed: {summary['batches_processed']}")
        print(f"{'='*60}\n")
        
        # Save summary
        summary_file = self.checkpoint_dir / "dsl_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="DSL-Based Self-Regulated Pipeline for FinQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/hsin_research/FinQA-main/dataset/train.json",
        help="Path to the FinQA dataset JSON file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples per batch"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process"
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="Starting batch number"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/checkpoints/dsl",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--seed-rulebook",
        type=str,
        default="data/dsl_seed_rules.xml",
        help="Path to seed rulebook XML file"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed:
        random.seed(args.seed)
    
    # Initialize components
    print("Initializing DSL pipeline components...")
    
    data_buffer = DataBuffer(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )
    
    solver = DSLSolverAgent()
    optimizer = OptimizerAgent()
    evaluator = DSLEvaluator(tolerance=0.01)
    
    pipeline = DSLPipeline(
        solver=solver,
        optimizer=optimizer,
        evaluator=evaluator,
        checkpoint_dir=args.checkpoint_dir,
        seed_rulebook_path=args.seed_rulebook
    )
    
    # Run pipeline
    summary = pipeline.run(
        data_buffer=data_buffer,
        max_batches=args.max_batches,
        start_batch=args.start_batch
    )
    
    print("✅ DSL Pipeline complete!")
    return summary


if __name__ == "__main__":
    main()
