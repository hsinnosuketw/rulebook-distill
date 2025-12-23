import json
import os
import re
import math
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm
from prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

def load_finqa_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def str_to_num(text):
    if text is None:
        return "n/a"
    text = str(text).replace(",", "").replace("$", "").strip()
    try:
        num = float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                num = float(text)
                num = num / 100.0
            except ValueError:
                num = "n/a"
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            try:
                num = float(text)
            except:
                num = "n/a"
        else:
            num = "n/a"
    return num

def clean_answer_robust(ans_str):
    """Robustly extracts numerical value from response string."""
    if not ans_str or ans_str == "n/a":
        return "n/a"
    
    # Pre-processing: remove $, commas, and handle parenthetical negative numbers common in finance
    temp_str = ans_str.replace(",", "").replace("$", "").strip()
    if temp_str.startswith("(") and temp_str.endswith(")"):
        inner = temp_str[1:-1].strip()
        if re.match(r"^\d+(?:\.\d+)?%?$", inner):
            temp_str = "-" + inner

    # If it's a formula or has '=', prioritize the part after the last '='
    if '=' in temp_str:
        temp_str = temp_str.split('=')[-1].strip()
    
    # Regex to find numbers: handles integers, floats, and percentages
    pattern = r"-?\d+(?:\.\d+)?%?"
    matches = re.findall(pattern, temp_str)
    
    if not matches:
        return "n/a"
    
    # Heuristic: Take the last number in the selected part of the string
    last_match = matches[-1]
    
    is_percent = False
    if last_match.endswith("%"):
        is_percent = True
        last_match = last_match[:-1]
    
    try:
        val = float(last_match)
        if is_percent:
            val = val / 100.0
        return val
    except:
        return "n/a"

def eval_program(program_str):
    try:
        if not program_str or program_str == "n/a":
            return "n/a"
        steps = program_str.split("),")
        res_dict = {}
        this_res = "n/a"
        
        for ind, step in enumerate(steps):
            step = step.strip()
            if not step.endswith(")") and ind == len(steps) - 1:
                step += ")"
            
            if "(" not in step or ")" not in step:
                continue
                
            op = step.split("(")[0].strip()
            args_str = step.split("(")[1].split(")")[0].strip()
            args = [arg.strip() for arg in args_str.split(",")]
            
            if len(args) < 2:
                continue
                
            arg1_str, arg2_str = args[0], args[1]
            
            if "#" in arg1_str:
                arg1 = res_dict[int(arg1_str.replace("#", ""))]
            else:
                arg1 = str_to_num(arg1_str)
                
            if "#" in arg2_str:
                arg2 = res_dict[int(arg2_str.replace("#", ""))]
            else:
                arg2 = str_to_num(arg2_str)
                
            if arg1 == "n/a" or arg2 == "n/a":
                return "n/a"
                
            if op == "add":
                this_res = arg1 + arg2
            elif op == "subtract":
                this_res = arg1 - arg2
            elif op == "multiply":
                this_res = arg1 * arg2
            elif op == "divide":
                this_res = arg1 / arg2 if arg2 != 0 else 0
            elif op == "exp":
                this_res = math.pow(arg1, arg2)
            elif op == "greater":
                this_res = 1.0 if arg1 > arg2 else 0.0
            else:
                this_res = "n/a"
                
            res_dict[ind] = this_res
        return this_res
    except Exception:
        return "n/a"

def classify_error(pred_val, gt_val, raw_response, tolerance=1e-4, rel_tolerance=0.01):
    if gt_val == "n/a" or gt_val is None:
        return "unknown (missing gt)"
    if pred_val == "n/a":
        return "computation error (null prediction)"
        
    is_correct = False
    try:
        if abs(gt_val - pred_val) < tolerance or (gt_val != 0 and abs((gt_val - pred_val) / gt_val) < rel_tolerance):
            is_correct = True
    except:
        pass
        
    if is_correct:
        return "correct"
    
    # Classification logic
    has_formula = any(op in raw_response for op in ['+', '*', '/', '='])
    relative_diff = abs(gt_val - pred_val) / (abs(gt_val) + 1e-9)
    
    if has_formula:
        return "representation error (formulaic output)"
    if relative_diff < 0.05: 
        return "representation error (near match/rounding)"
        
    return "computation error"

def get_direct_answer(question, context):
    formatted_user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_user_prompt}
            ],
            temperature=0.0,
            max_tokens=128 # Increased from 20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return ""

def evaluate_accuracy(dataset, limit=None, output_file="results.jsonl"):
    correct_count = 0
    total_count = 0
    
    samples = dataset[:limit] if limit else dataset
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(samples, desc="Evaluating"):
            qa = item['qa']
            question = qa['question']
            context = " ".join(qa['gold_inds'].values())
            program = qa.get('program', '')
            
            gt_val = eval_program(program)
            if gt_val == "n/a":
                gt_val = str_to_num(str(qa.get('answer', '')))

            raw_pred = get_direct_answer(question, context)
            pred_val = clean_answer_robust(raw_pred)
            
            error_cat = classify_error(pred_val, gt_val, raw_pred)
            is_correct = (error_cat == "correct")
            
            result = {
                "question": question,
                "dataset_answer": qa.get('answer'),
                "program": program,
                "program_ground_truth": gt_val,
                "raw_response": raw_pred,
                "parsed_prediction": pred_val,
                "error_category": error_cat,
                "is_correct": is_correct
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            
            if is_correct:
                correct_count += 1
            total_count += 1
        
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, correct_count, total_count

if __name__ == "__main__":
    dataset_path = "/root/hsin_research/FinQA-main/dataset/train.json"
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_finqa_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples.")
    
    limit = None 
    print(f"Starting evaluation (limit={limit}, max_tokens=128)...")
    
    accuracy, correct, total = evaluate_accuracy(dataset, limit=limit)
    
    print("\n--- Results ---")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
