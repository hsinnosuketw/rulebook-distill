"""
Neuro-Symbolic DSL Pipeline

EXTENDS the original DSL pipeline with neuro-symbolic features:
- Execution trace feedback (CodePRM approach)
- Wake-Sleep library learning (DreamCoder approach)
- Sketch-based rules

This is a DROP-IN replacement for dsl_pipeline.py that adds neuro-symbolic enhancements.
Set NEURO_SYMBOLIC_ENABLED = False to use original behavior.

Usage:
    python neuro_dsl_pipeline.py --dataset /path/to/train.json --batch-size 10 --neuro
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_buffer import DataBuffer
from dsl_solver_agent import DSLSolverAgent
from dsl_evaluator import DSLEvaluator, compare_results, parse_program_from_string
from optimizer_agent import OptimizerAgent

# Neuro-symbolic extensions
from execution_tracer import eval_program_with_trace, TraceComparator, ExecutionTrace
from trace_buffer import TraceBuffer
from library_learner import LibraryLearner
from sketch_library import SketchLibrary

# Utilities
from utils.output_logger import OutputLogger

import config


class NeuroSymbolicDSLPipeline:
    """
    Extended DSL Pipeline with neuro-symbolic enhancements.
    
    Inherits all functionality from DSLPipeline and adds:
    1. Execution trace feedback for richer error diagnosis
    2. Wake-Sleep learning for automatic pattern abstraction
    3. Sketch-based rule integration
    
    Set neuro_enabled=False in __init__ to disable enhancements.
    """
    
    def __init__(
        self,
        solver: DSLSolverAgent,
        optimizer: OptimizerAgent,
        evaluator: DSLEvaluator,
        checkpoint_dir: str = "data/checkpoints/neuro_dsl",
        seed_rulebook_path: str = None,
        # Output path options
        rulebook_dir: str = None,
        metrics_dir: str = None,
        # Neuro-symbolic options
        neuro_enabled: bool = True,
        sleep_interval: int = 50,
        min_cluster_size: int = 5,
        trace_buffer_size: int = 500
    ):
        """
        Initialize the neuro-symbolic DSL pipeline.
        
        Args:
            solver: DSL solver agent
            optimizer: Optimizer agent for rulebook updates  
            evaluator: DSL program evaluator
            checkpoint_dir: Directory for saving checkpoints (results)
            seed_rulebook_path: Optional path to seed rulebook XML
            rulebook_dir: Custom directory for rulebook outputs (defaults to checkpoint_dir)
            metrics_dir: Custom directory for metrics outputs (defaults to checkpoint_dir)
            neuro_enabled: Enable neuro-symbolic enhancements
            sleep_interval: Run sleep phase every N batches
            min_cluster_size: Minimum traces for pattern abstraction
            trace_buffer_size: Maximum traces to buffer
        """
        self.solver = solver
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate output directories
        self.rulebook_dir = Path(rulebook_dir) if rulebook_dir else self.checkpoint_dir
        self.rulebook_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_dir = Path(metrics_dir) if metrics_dir else self.checkpoint_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Output logger for model predictions
        self.output_logger = OutputLogger(
            output_path=str(self.checkpoint_dir / "model_outputs.jsonl")
        )
        
        # NEURO-SYMBOLIC EXTENSIONS
        self.neuro_enabled = neuro_enabled
        self.sleep_interval = sleep_interval
        
        if neuro_enabled:
            print("🧠 Neuro-Symbolic Mode: ENABLED")
            # Initialize trace buffer for wake-sleep learning
            buffer_path = self.checkpoint_dir / "trace_buffer.json"
            self.trace_buffer = TraceBuffer(
                max_traces=trace_buffer_size,
                checkpoint_path=str(buffer_path)
            )
            
            # Initialize library learner for sleep phase
            self.library_learner = LibraryLearner(
                min_cluster_size=min_cluster_size,
                max_new_rules=5
            )
            
            # Initialize trace comparator for error diagnosis
            self.trace_comparator = TraceComparator(tolerance=0.01)
            
            # Initialize PERSISTENT sketch library (separate from NL rules)
            sketch_path = self.checkpoint_dir / "sketch_library.json"
            self.sketch_library = SketchLibrary(
                checkpoint_path=str(sketch_path),
                max_sketches=10
            )
            print(f"   Sketch library: {len(self.sketch_library)} sketches loaded")
            
            # Track sleep phase
            self.last_sleep_batch = 0
        else:
            print("🔧 Neuro-Symbolic Mode: DISABLED (using original pipeline)")
            self.trace_buffer = None
            self.library_learner = None
            self.trace_comparator = None
            self.sketch_library = None
    
    def _save_results(self, results: list, batch_num: int):
        """Save batch results to JSONL file."""
        results_file = self.checkpoint_dir / f"dsl_results_batch_{batch_num:03d}.jsonl"
        with open(results_file, "w") as f:
            for result in results:
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
                    "error": result.get("error"),
                    # Neuro-symbolic additions
                    "trace_diagnosis": result.get("trace_diagnosis") if self.neuro_enabled else None
                }
                f.write(json.dumps(serializable) + "\n")
    
    def _save_rulebook(self, batch_num: int, suffix: str = ""):
        """Save current rulebook."""
        rulebook_file = self.rulebook_dir / f"dsl_rulebook_batch_{batch_num:03d}{suffix}.xml"
        with open(rulebook_file, "w") as f:
            f.write(self.rulebook)
    
    def _save_metrics(self, metrics: dict):
        """Append metrics to log file."""
        metrics_file = self.metrics_dir / "dsl_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def _compute_trace_diagnosis(
        self, 
        result: dict
    ) -> Optional[dict]:
        """
        Compute execution trace diagnosis for a failed result.
        
        This is the CodePRM-style feedback mechanism.
        """
        if not self.neuro_enabled or not self.trace_comparator:
            return None
        
        if result.get("is_correct"):
            return None  # No diagnosis needed for correct answers
        
        try:
            # Get model program
            model_prog = result.get("program", [])
            if isinstance(model_prog, str):
                model_prog = parse_program_from_string(model_prog)
            
            # Get GT program
            gt_prog = result.get("gt_program", "")
            if isinstance(gt_prog, str):
                gt_prog = parse_program_from_string(gt_prog)
            
            # Compute traces
            model_trace = eval_program_with_trace(model_prog)
            gt_trace = eval_program_with_trace(gt_prog)
            
            # Diff traces
            diagnosis = self.trace_comparator.diff_traces(
                model_trace, 
                gt_trace,
                question=result.get("question", "")
            )
            
            return diagnosis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _format_failures_with_traces(self, results: list) -> list:
        """
        Format failed results with trace-based diagnoses for optimizer.
        
        This provides richer feedback than just predicted vs expected.
        """
        failures = []
        for result in results:
            if not result.get("is_correct", False):
                model_prog = result.get("program", [])
                if isinstance(model_prog, list):
                    model_prog_str = " ".join(str(t) for t in model_prog if t != "EOF")
                else:
                    model_prog_str = str(model_prog)
                
                failure_entry = {
                    "idx": result.get("idx"),
                    "question": str(result.get("question", "")),
                    "predicted": str(result.get("result")) if result.get("result") is not None else "EXECUTION_ERROR",
                    "ground_truth": str(result.get("gt_result", "")),
                    "reasoning": f"Model program: {model_prog_str}",
                    "rules_applied": [],
                    "error": result.get("error", "Result mismatch"),
                    "gt_program": str(result.get("gt_program", ""))
                }
                
                # Add trace diagnosis if available
                if self.neuro_enabled and result.get("trace_diagnosis"):
                    diag = result["trace_diagnosis"]
                    failure_entry["trace_diagnosis"] = diag.get("diagnosis", "")
                    failure_entry["error_type"] = diag.get("error_type", "UNKNOWN")
                    failure_entry["corrective_hint"] = diag.get("corrective_hint", "")
                
                failures.append(failure_entry)
        
        return failures
    
    def _wake_phase(self, solver_results: list, batch_num: int):
        """
        Wake Phase: Store successful program traces.
        
        Only runs if neuro_enabled is True.
        """
        if not self.neuro_enabled or self.trace_buffer is None:
            return
        
        for result in solver_results:
            if result.get("is_correct") and result.get("success"):
                self.trace_buffer.add_success(
                    question=result.get("question", ""),
                    program=result.get("program", []),
                    result=result.get("result"),
                    metadata={"batch": batch_num}
                )
        
        # Save buffer checkpoint periodically
        if batch_num % 10 == 0:
            self.trace_buffer.save_checkpoint()
    
    def _sleep_phase(self, batch_num: int):
        """
        Sleep Phase: Abstract patterns from successful traces.
        
        Only runs every sleep_interval batches if neuro_enabled is True.
        """
        if not self.neuro_enabled or self.library_learner is None:
            return
        
        if batch_num - self.last_sleep_batch < self.sleep_interval:
            return
        
        print(f"\n💤 SLEEP PHASE (batch {batch_num})")
        print(f"   Buffer size: {len(self.trace_buffer)}")
        
        # Get abstraction stats
        stats = self.library_learner.get_abstraction_stats(self.trace_buffer)
        print(f"   Abstractable patterns: {stats['abstractable_patterns']}")
        
        if stats['abstractable_patterns'] > 0:
            # Run library learning
            updated_rulebook, new_rules = self.library_learner.run_sleep_phase(
                self.trace_buffer,
                self.rulebook
            )
            
            if new_rules:
                print(f"   ✨ Learned {len(new_rules)} new sketch rules:")
                for rule in new_rules:
                    print(f"      - {rule['id']}")
                
                # ADD SKETCHES TO PERSISTENT LIBRARY (not just rulebook!)
                if self.sketch_library is not None:
                    self.sketch_library.add_sketches_from_rules(new_rules)
                    self.sketch_library.save()
                    print(f"   📚 Sketch library now has {len(self.sketch_library)} sketches")
                
                self.rulebook = updated_rulebook
                self._save_rulebook(batch_num, suffix="_sleep")
        
        self.last_sleep_batch = batch_num
        print()
    
    def run_batch(self, batch: list, batch_num: int) -> dict:
        """
        Run a single batch through the pipeline.
        
        Enhanced with neuro-symbolic features when enabled.
        """
        # Stage 1: Generate DSL programs
        print("Stage 1: DSL Solver generating programs...")
        
        # NEURO-SYMBOLIC: Combine NL rulebook with sketch templates
        combined_rulebook = self.rulebook
        if self.neuro_enabled and self.sketch_library is not None and len(self.sketch_library) > 0:
            # Add sketch templates to the rulebook that gets passed to solver
            sketch_section = self.sketch_library.format_for_solver()
            combined_rulebook = self.rulebook + sketch_section
            print(f"   (Using {len(self.sketch_library)} sketches from library)")
        
        solver_results = self.solver.predict_batch(batch, combined_rulebook)
        
        # Log model outputs (question, answer, exe_ans, context, response, thinking)
        self.output_logger.log_batch(
            solver_results=solver_results,
            batch_items=batch,
            batch_num=batch_num,
        )
        
        # NEURO-SYMBOLIC: Compute trace diagnoses for failures
        if self.neuro_enabled:
            for result in solver_results:
                if not result.get("is_correct"):
                    result["trace_diagnosis"] = self._compute_trace_diagnosis(result)
        
        # Calculate metrics
        total = len(solver_results)
        correct = sum(1 for r in solver_results if r.get("is_correct", False))
        program_matches = sum(1 for r in solver_results if r.get("program_match", False))
        execution_errors = sum(1 for r in solver_results if not r.get("success", False))
        accuracy = correct / total if total > 0 else 0
        program_match_rate = program_matches / total if total > 0 else 0
        
        # Save results
        self._save_results(solver_results, batch_num)
        
        # NEURO-SYMBOLIC: Wake Phase - store successful traces
        self._wake_phase(solver_results, batch_num)
        
        # Stage 2: Analyze failures and update rulebook
        failures = self._format_failures_with_traces(solver_results)
        
        if failures and batch_num > 0:
            print("Stage 2: Optimizer analyzing failures...")
            try:
                optimizer_result = self.optimizer.optimize(
                    batch_results=solver_results,
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
        
        # NEURO-SYMBOLIC: Sleep Phase - abstract patterns periodically
        self._sleep_phase(batch_num)
        
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
            "timestamp": datetime.now().isoformat(),
            # Neuro-symbolic metrics
            "neuro_enabled": self.neuro_enabled,
            "trace_buffer_size": len(self.trace_buffer) if self.trace_buffer else 0
        }
        self._save_metrics(metrics)
        self.metrics_log.append(metrics)
        
        return metrics
    
    def run(
        self,
        data_buffer: DataBuffer,
        max_batches: Optional[int] = None,
        start_batch: int = 0,
        epochs: int = 1,
        train_samples: Optional[int] = None
    ) -> dict:
        """
        Run the full pipeline.
        
        Args:
            data_buffer: Data buffer with batches
            max_batches: Maximum batches to process per epoch
            start_batch: Starting batch number
            epochs: Number of training epochs (default: 1)
            train_samples: Specific number of samples for epoch training (default: all)
        """
        mode_str = "NEURO-SYMBOLIC" if self.neuro_enabled else "STANDARD"
        print(f"\n{'='*60}")
        print(f"{mode_str} DSL PIPELINE")
        print(f"{'='*60}")
        print(f"Total batches: {len(data_buffer)}")
        print(f"Batch size: {data_buffer.batch_size}")
        print(f"Epochs: {epochs}")
        if train_samples:
            print(f"Training samples: {train_samples} (reduces to {train_samples // data_buffer.batch_size} batches)")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        if self.rulebook_dir != self.checkpoint_dir:
            print(f"Rulebook dir: {self.rulebook_dir}")
        if self.metrics_dir != self.checkpoint_dir:
            print(f"Metrics dir: {self.metrics_dir}")
        if self.neuro_enabled:
            print(f"Sleep interval: every {self.sleep_interval} batches")
        print(f"{'='*60}\n")
        
        # Determine effective batch limit for epoch training
        effective_batches = max_batches
        if train_samples:
            # Convert train_samples to number of batches
            train_batches = (train_samples + data_buffer.batch_size - 1) // data_buffer.batch_size
            effective_batches = min(effective_batches, train_batches) if effective_batches else train_batches
            print(f"📚 Epoch Training Mode: {train_samples} samples = {train_batches} batches")
        
        total_correct = 0
        total_count = 0
        total_execution_errors = 0
        
        # Multi-epoch training loop
        for epoch in range(epochs):
            if epochs > 1:
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch + 1}/{epochs}")
                print(f"{'='*60}\n")
            
            epoch_correct = 0
            epoch_count = 0
            
            for batch_num, batch in enumerate(data_buffer):
                if batch_num < start_batch:
                    continue
                
                if effective_batches and batch_num >= start_batch + effective_batches:
                    break
                
                # Adjust batch number for multi-epoch (global batch counter)
                global_batch_num = start_batch + (epoch * (effective_batches or len(data_buffer))) + (batch_num - start_batch)
                
                print(f"\n--- Epoch {epoch+1}/{epochs}, Batch {batch_num}/{len(data_buffer)} (Global #{global_batch_num}) ---")
                
                metrics = self.run_batch(batch, global_batch_num)
                
                epoch_count += metrics["total_count"]
                epoch_correct += metrics["correct_count"]
                total_count += metrics["total_count"]
                total_correct += metrics["correct_count"]
                total_execution_errors += metrics["execution_errors"]
                
                print(f"Exec Accuracy: {metrics['accuracy']*100:.1f}% ({metrics['correct_count']}/{metrics['total_count']})")
                print(f"Program Match: {metrics['program_match_rate']*100:.1f}% ({metrics['program_match_count']}/{metrics['total_count']})")
                print(f"Execution errors: {metrics['execution_errors']}")
                print(f"Rulebook size: {metrics['rulebook_size']} chars")
                if self.neuro_enabled:
                    print(f"Trace buffer: {metrics['trace_buffer_size']} traces")
            
            # Epoch summary
            if epochs > 1:
                epoch_acc = epoch_correct / epoch_count if epoch_count > 0 else 0
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch + 1} SUMMARY: {epoch_correct}/{epoch_count} = {epoch_acc*100:.1f}%")
                print(f"{'='*60}")
        
        # Final summary
        overall_accuracy = total_correct / total_count if total_count > 0 else 0
        
        summary = {
            "total_count": total_count,
            "total_correct": total_correct,
            "execution_errors": total_execution_errors,
            "overall_accuracy": overall_accuracy,
            "batches_processed": len(self.metrics_log),
            "neuro_enabled": self.neuro_enabled
        }
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {total_count}")
        print(f"Correct: {total_correct} ({overall_accuracy*100:.2f}%)")
        print(f"Execution errors: {total_execution_errors}")
        print(f"Batches processed: {summary['batches_processed']}")
        if self.neuro_enabled and self.trace_buffer:
            print(f"Final trace buffer: {len(self.trace_buffer)} traces")
        print(f"{'='*60}\n")
        
        # Save summary
        summary_file = self.metrics_dir / "dsl_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save final trace buffer
        if self.neuro_enabled and self.trace_buffer:
            self.trace_buffer.save_checkpoint()
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic DSL Pipeline for FinQA",
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
        default="data/checkpoints/neuro_dsl",
        help="Directory for saving batch results and checkpoints"
    )
    parser.add_argument(
        "--rulebook-dir",
        type=str,
        default=None,
        help="Custom directory for rulebook outputs (defaults to checkpoint-dir)"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Custom directory for metrics outputs (defaults to checkpoint-dir)"
    )
    parser.add_argument(
        "--seed-rulebook",
        type=str,
        default="data/dsl_seed_rules.xml",
        help="Path to seed rulebook XML file"
    )
    # Neuro-symbolic options
    parser.add_argument(
        "--neuro",
        action="store_true",
        help="Enable neuro-symbolic enhancements"
    )
    parser.add_argument(
        "--sleep-interval",
        type=int,
        default=50,
        help="Run sleep phase every N batches"
    )
    parser.add_argument(
        "--min-cluster",
        type=int,
        default=5,
        help="Minimum traces for pattern abstraction"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=500,
        help="Maximum traces in buffer"
    )
    # Epoch training options
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Number of samples to use for epoch training (uses all if not specified)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs to run on the selected samples"
    )
    # Rule selection options
    parser.add_argument(
        "--enable-rule-selection",
        action="store_true",
        help="Enable BM25-based rule selection to reduce noise from injecting all rules"
    )
    parser.add_argument(
        "--top-k-rules",
        type=int,
        default=3,
        help="Number of top rules to select when rule selection is enabled"
    )
    # LLM Backend options
    parser.add_argument(
        "--backend",
        type=str,
        choices=["nvidia", "ollama"],
        default="ollama",
        help="LLM backend to use: 'nvidia' for NVIDIA API or 'ollama' for local Ollama (default: from config or env LLM_BACKEND)"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Model name for Ollama backend (e.g., 'qwen3-next:latest', default: from config or env OLLAMA_MODEL)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed:
        random.seed(args.seed)
    
    # Initialize components
    print("Initializing Neuro-Symbolic DSL pipeline components...")
    
    data_buffer = DataBuffer(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle
    )
    
    solver = DSLSolverAgent(
        enable_rule_selection=args.enable_rule_selection,
        top_k_rules=args.top_k_rules,
        backend=args.backend,
        ollama_model=args.ollama_model
    )
    optimizer = OptimizerAgent(
        backend=args.backend,
        ollama_model=args.ollama_model
    )
    evaluator = DSLEvaluator(tolerance=0.01)
    
    pipeline = NeuroSymbolicDSLPipeline(
        solver=solver,
        optimizer=optimizer,
        evaluator=evaluator,
        checkpoint_dir=args.checkpoint_dir,
        seed_rulebook_path=args.seed_rulebook,
        rulebook_dir=args.rulebook_dir,
        metrics_dir=args.metrics_dir,
        neuro_enabled=args.neuro,
        sleep_interval=args.sleep_interval,
        min_cluster_size=args.min_cluster,
        trace_buffer_size=args.buffer_size
    )
    
    # Run pipeline
    summary = pipeline.run(
        data_buffer=data_buffer,
        max_batches=args.max_batches,
        start_batch=args.start_batch,
        epochs=args.epochs,
        train_samples=args.train_samples
    )
    
    print("✅ Neuro-Symbolic DSL Pipeline complete!")
    return summary


if __name__ == "__main__":
    main()
