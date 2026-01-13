"""
Self-Regulated Rule Augmentation Pipeline

A closed-loop feedback system where the Rulebook dynamically evolves
based on Solver failures. The Optimizer Agent analyzes errors and
synthesizes new/refined rules to prevent future mistakes.

Usage:
    python self_regulated_pipeline.py --dataset /path/to/train.json --batch-size 10 --max-batches 5
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional

from data_buffer import DataBuffer
from solver_agent import SolverAgent
from optimizer_agent import OptimizerAgent
from rulebook_utils import (
    get_empty_rulebook,
    count_rules,
    parse_rulebook,
    serialize_rulebook,
    validate_rulebook
)


class SelfRegulatedPipeline:
    """
    Orchestrates the self-regulating rule augmentation loop.
    
    Components:
    - DataBuffer: Manages batch iteration over dataset
    - SolverAgent: Executes CoT reasoning with current rulebook
    - OptimizerAgent: Analyzes failures and evolves rulebook
    
    The pipeline runs in batches:
    1. Solver answers N questions using current rulebook
    2. Optimizer analyzes failures and updates rulebook
    3. Repeat with new rulebook
    """
    
    def __init__(
        self,
        solver: SolverAgent,
        optimizer: OptimizerAgent,
        max_rules: int = 15,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            solver: SolverAgent instance
            optimizer: OptimizerAgent instance
            max_rules: Maximum number of rules in the rulebook
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print progress
        """
        self.solver = solver
        self.optimizer = optimizer
        self.max_rules = max_rules
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        
        # Initialize empty rulebook
        self.rulebook = get_empty_rulebook()
        
        # History tracking
        self.history = []
        self.metrics_history = []
    
    def run(
        self,
        data_buffer: DataBuffer,
        max_batches: Optional[int] = None,
        early_stop_accuracy: float = 0.95,
        early_stop_patience: int = 3
    ) -> dict:
        """
        Run the self-regulating pipeline.
        
        Args:
            data_buffer: DataBuffer instance for batch iteration
            max_batches: Maximum number of batches to process (None = all)
            early_stop_accuracy: Stop if accuracy exceeds this threshold
            early_stop_patience: Stop after N batches with no improvement
            
        Returns:
            Dictionary with final rulebook, history, and metrics
        """
        start_time = datetime.now()
        best_accuracy = 0.0
        patience_counter = 0
        batch_num = 0
        
        total_batches = min(max_batches, len(data_buffer)) if max_batches else len(data_buffer)
        
        self._log(f"\n{'='*60}")
        self._log(f"Self-Regulated Rule Augmentation Pipeline")
        self._log(f"{'='*60}")
        self._log(f"Total batches: {total_batches}")
        self._log(f"Batch size: {data_buffer.batch_size}")
        self._log(f"Max rules: {self.max_rules}")
        self._log(f"{'='*60}\n")
        
        for batch in data_buffer:
            if max_batches and batch_num >= max_batches:
                break
            
            self._log(f"\n--- Batch {batch_num + 1}/{total_batches} ---")
            
            # Stage 1: Solver Execution
            self._log("Stage 1: Solver executing predictions...")
            solver_results = self.solver.predict_batch(batch, self.rulebook)
            
            # Stage 2: Optimizer Analysis and Rulebook Update
            self._log("Stage 2: Optimizer analyzing failures...")
            optimization_result = self.optimizer.optimize(
                batch_results=solver_results,
                current_rulebook=self.rulebook,
                batch_num=batch_num,
                max_rules=self.max_rules
            )
            
            # Update rulebook
            old_rulebook = self.rulebook
            self.rulebook = optimization_result["new_rulebook"]
            
            # Record metrics
            metrics = optimization_result["metrics"]
            accuracy = metrics["accuracy"]
            rule_count = count_rules(self.rulebook)
            
            self._log(f"Accuracy: {accuracy:.1%} ({metrics['correct_count']}/{metrics['total_count']})")
            self._log(f"Failures: {metrics['error_count']}")
            self._log(f"Rule count: {rule_count}")
            
            # Track history
            batch_record = {
                "batch_num": batch_num,
                "accuracy": accuracy,
                "total": metrics["total_count"],
                "correct": metrics["correct_count"],
                "errors": metrics["error_count"],
                "rule_count": rule_count,
                "rulebook_changed": old_rulebook != self.rulebook,
                "timestamp": datetime.now().isoformat()
            }
            self.history.append(batch_record)
            self.metrics_history.append(metrics)
            
            # Save checkpoint
            if self.checkpoint_dir:
                self._save_checkpoint(batch_num, solver_results, optimization_result)
            
            # Early stopping checks
            if accuracy >= early_stop_accuracy:
                self._log(f"\n✓ Early stopping: Accuracy {accuracy:.1%} >= {early_stop_accuracy:.1%}")
                break
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    self._log(f"\n✓ Early stopping: No improvement for {early_stop_patience} batches")
                    break
            
            batch_num += 1
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        final_result = {
            "final_rulebook": self.rulebook,
            "history": self.history,
            "metrics_history": self.metrics_history,
            "total_batches_processed": batch_num + 1,
            "final_rule_count": count_rules(self.rulebook),
            "best_accuracy": best_accuracy,
            "duration_seconds": duration
        }
        
        self._print_summary(final_result)
        
        return final_result
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _save_checkpoint(
        self,
        batch_num: int,
        solver_results: list[dict],
        optimization_result: dict
    ):
        """Save checkpoint files for the current batch."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save current rulebook
        rulebook_path = os.path.join(self.checkpoint_dir, f"rulebook_batch_{batch_num:03d}.xml")
        with open(rulebook_path, 'w') as f:
            f.write(self.rulebook)
        
        # Save solver results
        results_path = os.path.join(self.checkpoint_dir, f"results_batch_{batch_num:03d}.jsonl")
        with open(results_path, 'w') as f:
            for result in solver_results:
                # Remove raw_item to avoid serialization issues
                result_copy = {k: v for k, v in result.items() if k != 'raw_item'}
                f.write(json.dumps(result_copy, ensure_ascii=False) + '\n')
        
        # Save metrics
        metrics_path = os.path.join(self.checkpoint_dir, "metrics.jsonl")
        with open(metrics_path, 'a') as f:
            metrics = {
                "batch_num": batch_num,
                **optimization_result["metrics"],
                "timestamp": datetime.now().isoformat()
            }
            f.write(json.dumps(metrics) + '\n')
    
    def _print_summary(self, result: dict):
        """Print final summary."""
        self._log(f"\n{'='*60}")
        self._log("Pipeline Complete")
        self._log(f"{'='*60}")
        self._log(f"Batches processed: {result['total_batches_processed']}")
        self._log(f"Best accuracy: {result['best_accuracy']:.1%}")
        self._log(f"Final rule count: {result['final_rule_count']}")
        self._log(f"Duration: {result['duration_seconds']:.1f} seconds")
        self._log(f"{'='*60}")
        
        # Print accuracy trend
        if len(self.history) > 1:
            self._log("\nAccuracy Trend:")
            for record in self.history:
                bar = "█" * int(record['accuracy'] * 20)
                self._log(f"  Batch {record['batch_num']:2d}: {bar} {record['accuracy']:.1%}")
    
    def get_final_rulebook(self) -> str:
        """Return the current rulebook."""
        return self.rulebook
    
    def get_rules_as_list(self) -> list[dict]:
        """Return current rules as a list of dictionaries."""
        return parse_rulebook(self.rulebook)
    
    def save_final_rulebook(self, path: str):
        """Save the final rulebook to a file."""
        with open(path, 'w') as f:
            f.write(self.rulebook)
        self._log(f"Rulebook saved to: {path}")
    
    def save_history(self, path: str):
        """Save the pipeline history to a JSON file."""
        with open(path, 'w') as f:
            json.dump({
                "history": self.history,
                "final_rulebook": self.rulebook
            }, f, indent=2)
        self._log(f"History saved to: {path}")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Self-Regulated Rule Augmentation Pipeline",
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
        help="Number of questions per batch"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (default: all)"
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=15,
        help="Maximum number of rules in the rulebook"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total dataset samples"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/root/hsin_research/ruledistill-main/data/checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--output-rulebook",
        type=str,
        default="/root/hsin_research/ruledistill-main/data/evolved_rulebook.xml",
        help="Path to save the final rulebook"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping entirely"
    )
    parser.add_argument(
        "--early-stop-accuracy",
        type=float,
        default=0.95,
        help="Stop if accuracy exceeds this threshold"
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop after N batches with no accuracy improvement"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    print("Initializing pipeline components...")
    
    data_buffer = DataBuffer(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        limit=args.limit,
        seed=args.seed
    )
    
    solver = SolverAgent(client_type="nvidia")
    optimizer = OptimizerAgent(client_type="nvidia")
    
    pipeline = SelfRegulatedPipeline(
        solver=solver,
        optimizer=optimizer,
        max_rules=args.max_rules,
        checkpoint_dir=args.checkpoint_dir,
        verbose=not args.quiet
    )
    
    # Run pipeline
    if args.no_early_stop:
        # Disable early stopping by setting unreachable thresholds
        result = pipeline.run(
            data_buffer=data_buffer,
            max_batches=args.max_batches,
            early_stop_accuracy=2.0,  # Impossible to reach
            early_stop_patience=999999  # Effectively infinite
        )
    else:
        result = pipeline.run(
            data_buffer=data_buffer,
            max_batches=args.max_batches,
            early_stop_accuracy=args.early_stop_accuracy,
            early_stop_patience=args.early_stop_patience
        )
    
    # Save outputs
    pipeline.save_final_rulebook(args.output_rulebook)
    
    history_path = os.path.join(args.checkpoint_dir, "pipeline_history.json")
    pipeline.save_history(history_path)
    
    print(f"\nFinal rulebook saved to: {args.output_rulebook}")
    print(f"Pipeline history saved to: {history_path}")


if __name__ == "__main__":
    main()
