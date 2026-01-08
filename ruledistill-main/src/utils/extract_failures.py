import json
import os

def extract():
    results_path = "/root/hsin_research/results.jsonl"
    output_path = "/root/hsin_research/failed_results_with_ids.jsonl"
    
    print("Processing results from results.jsonl...")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    failures = []
    count = 0
    with open(results_path, 'r') as f:
        # Enumerate to get the row number (0-indexed)
        # This row number corresponds directly to the index in train.json
        for idx, line in enumerate(f):
            entry = json.loads(line)
            if not entry.get('is_correct'):
                entry['id'] = idx  # The ID is now the row index
                failures.append(entry)
                count += 1
                
    with open(output_path, 'w') as f:
        for fail in failures:
            f.write(json.dumps(fail, ensure_ascii=False) + "\n")
            
    print(f"Extraction complete. {count} failures saved to {output_path}")
    print(f"Each failure now uses its row index from 'results.jsonl' as its 'id'.")

if __name__ == "__main__":
    extract()
