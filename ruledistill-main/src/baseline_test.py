"""
Baseline Test Evaluation Script

Evaluates the DSL solver with current evolved rulebook on the test set
to establish baseline performance before neuro-symbolic enhancements.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dsl_solver_agent import DSLSolverAgent
from dsl_evaluator import DSLEvaluator, compare_results


def load_test_data(dataset_path: str, max_samples: int = None) -> list:
    """Load and format test data."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    formatted = []
    for idx, item in enumerate(data):
        # Extract table from pre_text and post_text
        table = item.get('table', [])
        
        # Build context from pre_text + table + post_text
        pre_text = " ".join(item.get('pre_text', []))
        post_text = " ".join(item.get('post_text', []))
        
        # Format table as text for context
        table_text = ""
        if table:
            for row in table:
                table_text += " | ".join(str(cell) for cell in row) + "\n"
        
        context = f"{pre_text}\n\nTable:\n{table_text}\n{post_text}"
        
        # Get question and ground truth program
        qa_pair = item.get('qa', {})
        question = qa_pair.get('question', '')
        
        # Ground truth program
        gt_program = qa_pair.get('program', '')
        
        # Ground truth answer (exe_ans)
        gt_answer = qa_pair.get('exe_ans', '')
        
        formatted.append({
            'idx': idx,
            'question': question,
            'context': context,
            'table': table,
            'program': gt_program,  # GT program string
            'ground_truth': gt_answer
        })
    
    return formatted


def run_baseline_test(
    dataset_path: str,
    rulebook_path: str,
    output_path: str,
    max_samples: int = None,
    batch_size: int = 10
):
    """Run baseline evaluation on test set."""
    
    # Load rulebook
    rulebook = ""
    if rulebook_path and Path(rulebook_path).exists():
        with open(rulebook_path, 'r') as f:
            rulebook = f.read()
        print(f"Loaded rulebook: {len(rulebook)} chars")
    else:
        print("Running without rulebook (baseline)")
    
    # Load test data
    test_data = load_test_data(dataset_path, max_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # Initialize solver and evaluator
    solver = DSLSolverAgent(temperature=0.0)  # Deterministic
    evaluator = DSLEvaluator(tolerance=0.01)
    
    # Results storage
    all_results = []
    correct_count = 0
    program_match_count = 0
    execution_errors = 0
    
    # Process in batches
    total_batches = (len(test_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_data))
        batch = test_data[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")
        
        # Generate predictions
        batch_results = solver.predict_batch(batch, rulebook)
        
        for result in batch_results:
            all_results.append({
                'idx': result.get('idx'),
                'question': result.get('question', '')[:200],
                'model_program': result.get('program'),
                'gt_program': result.get('gt_program'),
                'model_result': result.get('result'),
                'gt_result': result.get('gt_result'),
                'is_correct': result.get('is_correct', False),
                'program_match': result.get('program_match', False),
                'success': result.get('success', False),
                'error': result.get('error')
            })
            
            if result.get('is_correct'):
                correct_count += 1
            if result.get('program_match'):
                program_match_count += 1
            if not result.get('success'):
                execution_errors += 1
        
        # Progress update
        current_accuracy = correct_count / len(all_results) * 100
        print(f"  Running accuracy: {current_accuracy:.1f}% ({correct_count}/{len(all_results)})")
    
    # Final summary
    total = len(all_results)
    accuracy = correct_count / total * 100 if total > 0 else 0
    program_match_rate = program_match_count / total * 100 if total > 0 else 0
    exec_error_rate = execution_errors / total * 100 if total > 0 else 0
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_path,
        'rulebook': rulebook_path,
        'total_samples': total,
        'correct': correct_count,
        'accuracy': accuracy,
        'program_matches': program_match_count,
        'program_match_rate': program_match_rate,
        'execution_errors': execution_errors,
        'execution_error_rate': exec_error_rate
    }
    
    print(f"\n{'='*60}")
    print("BASELINE TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Execution Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Program Match Rate: {program_match_rate:.2f}% ({program_match_count}/{total})")
    print(f"Execution Errors: {exec_error_rate:.2f}% ({execution_errors}/{total})")
    print(f"{'='*60}")
    
    # Save results
    output = {
        'summary': summary,
        'results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Test Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/hsin_research/FinQA-main/dataset/test.json",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--rulebook",
        type=str,
        default=None,
        help="Path to rulebook XML (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_test_results.json",
        help="Output file path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    
    run_baseline_test(
        dataset_path=args.dataset,
        rulebook_path=args.rulebook,
        output_path=args.output,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
