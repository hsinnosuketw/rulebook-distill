import json
import sys
import os

def inspect(row_index):
    dataset_path = "/root/hsin_research/FinQA-main/dataset/train.json"
    results_path = "/root/hsin_research/results.jsonl"
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    try:
        idx = int(row_index)
        if idx < 0 or idx >= len(dataset):
            print(f"Index {idx} is out of bounds (0-{len(dataset)-1}).")
            return
        target_data = dataset[idx]
    except ValueError:
        print(f"Error: Row index must be an integer. Received: {row_index}")
        return

    # Find in results (optimized lookup since we have the index)
    target_result = None
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            for i, line in enumerate(f):
                if i == idx:
                    target_result = json.loads(line)
                    break
    
    print("-" * 50)
    print(f"Row Index: {idx}")
    print(f"Question: {target_data['qa']['question']}")
    print("-" * 50)
    if target_result:
        print("EVALUATION RESULT (from results.jsonl):")
        print(f"  Raw Response: {target_result.get('raw_response')}")
        print(f"  Parsed Pred:  {target_result.get('parsed_prediction')}")
        print(f"  GT (Program): {target_result.get('program_ground_truth')}")
        print(f"  Is Correct:   {target_result.get('is_correct')}")
        print(f"  Error Cat:    {target_result.get('error_category')}")
    else:
        print(f"No evaluation result found for row index {idx} in results.jsonl.")
    
    print("-" * 50)
    print("DATASET INFO (from train.json):")
    print(f"  Original ID: {target_data.get('id')}")
    print(f"  Answer:      {target_data['qa'].get('answer')}")
    print(f"  Program:     {target_data['qa'].get('program')}")
    print(f"  Context:     {' '.join(target_data['qa']['gold_inds'].values())}")
    print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_sample.py <ROW_INDEX>")
        print("Example: python3 inspect_sample.py 0")
    else:
        inspect(sys.argv[1])
