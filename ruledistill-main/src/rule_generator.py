import json
import os
import sys
import re
import math
from dotenv import load_dotenv

# Import components from our existing evaluation pipeline
from src.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, SYSTEM_PROMPT_WITH_RULES_TEMPLATE
from src.evaluator import client, clean_answer_robust, eval_program, str_to_num, classify_error

# Try to import ollama, if not present, provide a dummy for script integrity
try:
    import ollama
except ImportError:
    ollama = None

def load_dataset(path="/root/hsin_research/FinQA-main/dataset/train.json"):
    print(f"Loading dataset from {path}...")
    with open(path, 'r') as f:
        return json.load(f)

def load_failures(path="/root/hsin_research/ruledistill-main/data/failed_results_with_ids.jsonl"):
    print(f"Loading failures from {path}...")
    failures = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                failures.append(json.loads(line))
    return failures

def get_rule_prompt(dataset_item, failure_item):
    question = dataset_item['qa']['question']
    context = " ".join(dataset_item['qa']['gold_inds'].values())
    ground_truth = dataset_item['qa'].get('answer')
    program = dataset_item['qa'].get('program')
    raw_response = failure_item.get('raw_response')
    parsed_prediction = failure_item.get('parsed_prediction')
    error_cat = failure_item.get('error_category')

    prompt = f"""As a professional Rule Extractor, analyze this financial reasoning failure and synthesize a rule to prevent it.

Question: {question}

Context: {context}

Ground Truth Answer: {ground_truth}
Ground Truth Program: {program}

Model Response: "{raw_response}"
Parsed Prediction: {parsed_prediction}
Error Category: {error_cat}

Task:
1. Perform Gap Analysis: Why did the model fail? (e.g., missed context, wrong sign, unit mismatch, formula output)
2. Synthesize a Rule: Draft a clear, direct rule in XML format:
<Rule id="[ID]" phase="generation" confidence="1" source="failure_log">
    <Trigger>[Specific keywords or task types]</Trigger>
    <Action>[Direct instruction on what to do or avoid]</Action>
</Rule>
3. Return the XML rule only.
"""
    return prompt

def generate_rules_for_range(start_idx, end_idx, model="gemini-3-pro-preview:latest"):
    if ollama is None:
        print("Error: 'ollama' package not found. Please install it with 'pip install ollama'.")
        return

    dataset = load_dataset()
    failures = load_failures()
    
    selected_failures = failures[start_idx:end_idx+1]
    print(f"Processing failures in 'failed_results_with_ids.jsonl' slice [{start_idx}:{end_idx+1}]...")
    
    rules = []
    for i, fail in enumerate(selected_failures):
        row_idx = fail.get('id')
        
        if row_idx is None or not isinstance(row_idx, int):
            continue
            
        dataset_item = dataset[row_idx]
        prompt = get_rule_prompt(dataset_item, fail)
        print(f"[{i+start_idx}] Generating rule for row index {row_idx}...")
        
        try:
            response = ollama.chat(model=model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            rules.append({
                "row_index": row_idx,
                "rule": response['message']['content'],
                "original_failure": fail
            })
        except Exception as e:
            print(f"Error generating rule for row index {row_idx}: {e}")
            
    return rules

def verify_rule_with_llama(dataset_item, rule_text):
    """
    Prompts Llama 3.3 70B with the new rule to see if it fixes the problem.
    """
    question = dataset_item['qa']['question']
    context = " ".join(dataset_item['qa']['gold_inds'].values())
    program = dataset_item['qa'].get('program', '')
    
    gt_val = eval_program(program)
    if gt_val == "n/a":
        gt_val = str_to_num(str(dataset_item['qa'].get('answer', '')))
    
    # Extract rule block if model included extra text
    rule_match = re.search(r'<Rule.*?>.*?</Rule>', rule_text, re.DOTALL)
    extracted_rule = rule_match.group(0) if rule_match else rule_text

    system_content = SYSTEM_PROMPT_WITH_RULES_TEMPLATE.format(rules=extracted_rule)
    user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=128
        )
        raw_pred = response.choices[0].message.content.strip()
        pred_val = clean_answer_robust(raw_pred)
        error_cat = classify_error(pred_val, gt_val, raw_pred)
        is_correct = (error_cat == "correct")
        
        return {
            "raw_response": raw_pred,
            "parsed_prediction": pred_val,
            "error_category": error_cat,
            "is_correct": is_correct,
            "ground_truth": gt_val
        }
    except Exception as e:
        print(f"Error verifying rule with Llama: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 rule_generator.py <start_idx> <end_idx> [model_name]")
    else:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        model = sys.argv[3] if len(sys.argv) > 3 else "gemini-3-pro-preview:latest"
        
        results = generate_rules_for_range(start, end, model=model)
        # For simplicity in CLI, we don't automatically verify here unless requested
        if results:
            output_file = f"generated_rules_{start}_{end}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Rules saved to {output_file}")
