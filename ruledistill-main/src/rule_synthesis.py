import json
import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
# Assuming the script is run from project root or we specify absolute path
load_dotenv("/root/hsin_research/.env")

# Paths
DATASET_PATH = "/root/hsin_research/FinQA-main/dataset/train.json"
FAILURES_PATH = "/root/hsin_research/ruledistill-main/data/failed_results_with_ids.jsonl"
OUTPUT_FILE = "synthesized_rules.json"
# Exact model name provided by user
MODEL_NAME = "deepseek-ai/deepseek-v3.2"

def load_dataset(path=DATASET_PATH):
    print(f"Loading dataset from {path}...")
    with open(path, 'r') as f:
        return json.load(f)

def load_failures(path=FAILURES_PATH):
    print(f"Loading failures from {path}...")
    failures = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                failures.append(json.loads(line))
    return failures

def get_gold_context(dataset_item):
    """
    Extracts the gold context from the dataset item.
    Assumes dataset_item['qa']['gold_inds'] contains the context sentences.
    """
    gold_inds = dataset_item.get('qa', {}).get('gold_inds', {})
    if isinstance(gold_inds, dict):
        return " ".join(gold_inds.values())
    elif isinstance(gold_inds, list):
        # Fallback if list: join elements if string
        return " ".join([str(x) for x in gold_inds])
    return str(gold_inds)

def construct_prompt(dataset_item, failure_item, sequential_id):
    question = dataset_item['qa']['question']
    context = get_gold_context(dataset_item)
    ground_truth = dataset_item['qa'].get('answer', 'N/A')
    
    # Determine model response
    if 'raw_response' in failure_item:
        model_response = failure_item['raw_response']
    elif 'parsed_prediction' in failure_item:
        model_response = str(failure_item['parsed_prediction'])
    else:
        model_response = "No response recorded."

    # Sequential ID formatting (e.g., 01, 02)
    seq_id_str = f"{sequential_id:02d}"

    prompt = f"""You are an expert AI Alignment Engineer specializing in error analysis and rule synthesis for financial reasoning models (FinQA).

Input Data:

Question: {question}

Context: {context}

Model's Incorrect Response: {model_response}

Correct Programmed Answer: {ground_truth}

Objective:

Analyze each failure and generate a General Rule to prevent recurrence.

Constraints & Guidelines:

Generalization is Key: The rule must be abstract and applicable to future, unseen cases.

NO DATA LEAKAGE: Do not include specific numbers, entity names, or result values from the specific test case in the rule. The rule must focus on the methodology or logic, not the specific content.

Bad Rule: "Do not calculate 50 + 20."

Good Rule: "Ensure that all addends mentioned in the 'Expenses' section are summed before calculating the net profit."

Strict XML Output: The output must strictly follow the schema provided below.

Workflow:

For each entry, perform the following steps:

Gap Analysis: Compare the ground_truth against the model_response. Identify if the error is due to arithmetic hallucination, program logic failure, or incorrect context retrieval.

Rule Synthesis: Draft a rule that addresses the root cause identified in Step 1.

Formatting: Encapsulate the rule in the XML block.

XML Output Template:

<Rulebook domain="finqa_reasoning">

    <Rule id="{seq_id_str}" phase="generation" confidence="1" source="log_{seq_id_str}" type="ErrorType">

        <Trigger>[Describe the scenario or keywords that trigger this rule]</Trigger>

        <Action>[The generalized instruction. Use prefixes: CRITICAL FORMATTING RULE or KNOWLEDGE INJECTION]</Action>

    </Rule></Rulebook>

The model must give a type to each error that the responding model made.
"""
    return prompt

def main():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable not set. Please check /root/hsin_research/.env")
        # Proceeding is futile without API key if the client needs it.
        # However, proceed to try block just in case env is set differently? 
        # No, let's just let the client init fail naturally if key is missing or prompt user.
    
    if not api_key:
        # Just fail gracefully with a clearer message here before client init
        sys.exit(1)

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    try:
        dataset = load_dataset()
        failures = load_failures()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load data. {e}")
        sys.exit(1)
        
    results = []
    
    limit = len(failures)
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass
            
    print(f"Processing {min(limit, len(failures))} failures out of {len(failures)} total.")

    for i, fail in enumerate(failures):
        if i >= limit:
            break
            
        row_id = fail.get('id')
        if row_id is None:
            print(f"Skipping item {i} due to missing 'id'")
            continue
            
        dataset_item = dataset[row_id]
        
        seq_id = i + 1 
        
        prompt = construct_prompt(dataset_item, fail, seq_id)
        
        print(f"[{i+1}/{limit}] Synthesizing rule for Dataset ID {row_id}...")
        
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                top_p=0.95,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )
            
            reasoning_accumulator = ""
            content_accumulator = ""
            
            print(f"  > Streaming response...", end="", flush=True)
            
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                
                # Capture reasoning if available
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'reasoning_content'):
                     delta_reasoning = getattr(choice.delta, "reasoning_content", None)
                     if delta_reasoning:
                        reasoning_accumulator += delta_reasoning
                    
                # Capture standard content
                if hasattr(choice, 'delta') and choice.delta.content:
                    delta_content = choice.delta.content
                    content_accumulator += delta_content
            
            print(" Done.")
            
            results.append({
                "dataset_id": row_id,
                "input_question": dataset_item['qa']['question'],
                "reasoning": reasoning_accumulator,
                "generated_rule_response": content_accumulator
            })
            
        except Exception as e:
            print(f"Error generating rule for ID {row_id}: {e}")
            continue

    # Save output
    output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Analysis complete. Synthesized rules saved to {output_path}")

if __name__ == "__main__":
    main()
