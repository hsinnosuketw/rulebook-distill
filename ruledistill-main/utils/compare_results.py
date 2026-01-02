import json
from collections import Counter

def compare_jsonl(baseline_path, rule_based_path):
    with open(baseline_path, 'r') as f:
        baseline = [json.loads(line) for line in f]
    with open(rule_based_path, 'r') as f:
        rule_based = [json.loads(line) for line in f]

    total = len(baseline)
    baseline_correct = sum(1 for x in baseline if x['is_correct'])
    rule_based_correct = sum(1 for x in rule_based if x['is_correct'])

    print(f"--- Global Statistics ---")
    print(f"Total Samples: {total}")
    print(f"Baseline Accuracy: {baseline_correct/total*100:.2f}% ({baseline_correct}/{total})")
    print(f"Rule-Based Accuracy: {rule_based_correct/total*100:.2f}% ({rule_based_correct}/{total})")
    print(f"Delta: {(rule_based_correct - baseline_correct)/total*100:+.2f}%")

    # Improvements vs Regressions
    improved = []
    regressed = []
    both_correct = 0
    both_wrong = 0

    for b, r in zip(baseline, rule_based):
        if not b['is_correct'] and r['is_correct']:
            improved.append((b, r))
        elif b['is_correct'] and not r['is_correct']:
            regressed.append((b, r))
        elif b['is_correct'] and r['is_correct']:
            both_correct += 1
        else:
            both_wrong += 1

    print(f"\n--- Detailed Comparison ---")
    print(f"Improved (Baseline Wrong -> Rule Correct): {len(improved)}")
    print(f"Regressed (Baseline Correct -> Rule Wrong): {len(regressed)}")
    print(f"Both Correct: {both_correct}")
    print(f"Both Wrong: {both_wrong}")

    # Breakdown by error category in baseline
    baseline_errors = Counter([x['error_category'] for x in baseline if not x['is_correct']])
    print(f"\n--- Baseline Error Breakdown ---")
    for cat, count in baseline_errors.items():
        print(f"{cat}: {count}")

    # Show some examples of improvement
    if improved:
        print(f"\n--- Examples of Improvement ---")
        for i in range(min(5, len(improved))):
            b, r = improved[i]
            print(f"Q: {b['question']}")
            print(f"GT: {b['program_ground_truth']}")
            print(f"Baseline Raw: {b['raw_response']} -> Parsed: {b['parsed_prediction']} ({b['error_category']})")
            print(f"Rule-Based Raw: {r['raw_response']} -> Parsed: {r['parsed_prediction']} ({r['error_category']})")
            print("-" * 20)

    # Show some examples of regression
    if regressed:
        print(f"\n--- Examples of Regression ---")
        for i in range(min(5, len(regressed))):
            b, r = regressed[i]
            print(f"Q: {b['question']}")
            print(f"GT: {b['program_ground_truth']}")
            print(f"Baseline: {b['raw_response']}")
            print(f"Rule-Based: {r['raw_response']}")
            print("-" * 20)

if __name__ == "__main__":
    compare_jsonl("/root/hsin_research/results.jsonl", "/root/hsin_research/results_with_rule.jsonl")
