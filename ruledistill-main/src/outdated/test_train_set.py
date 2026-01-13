import json
import judger
import model
from rule_search import RuleSearch
from tqdm import tqdm



if __name__ == "__main__":
    # Failed Data from failed_results_with_ids.jsonl
    test_qa_data = []
    with open('/root/hsin_research/FinQA-main/dataset/train.json','r') as f:
        test_qa_data = json.load(f)
    print(test_qa_data[0]['qa'].keys())

    # Load Judger, Model, Rule retriever
    judger = judger.Judger(client_type="nvidia")
    model = model.Model(client_type="nvidia")

    # Comment for removing the rule retriever
    # xml_path = '/root/hsin_research/ruledistill-main/data/all_rules.xml'
    # rule_retriever_bm25 = RuleSearch(xml_path)

    # Test rule retriever's performance on guiding the wrongly answered questions
    # print(original_qa_data[curr_num]['qa']['gold_inds'])
    for curr_num in tqdm(range(len(test_qa_data))):


        q_id = curr_num
        question = test_qa_data[q_id]['qa']['question']

        gold_index = test_qa_data[q_id]['qa']['gold_inds']
        # Comment for removing the rule retriever
        # retrieved_rules = rule_retriever_bm25.search(question, top_k=2)
        # context = [gold_index, retrieved_rules]
        context = gold_index
        ground_truth = test_qa_data[q_id]['qa']['answer']


        response = model.answer(context,question)
        evaluation = judger.evaluate(question, ground_truth, response)
        try:
            # Extract Final Evaluation
            if "# Final Evaluation" in evaluation:
                final_eval_str = evaluation.split("# Final Evaluation")[-1].strip()
                if "True" in final_eval_str:
                    evaluation_result = True
                elif "False" in final_eval_str:
                    evaluation_result = False
                else:
                    evaluation_result = None # Could not parse
            else:
                 evaluation_result = None

        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            evaluation_result = None

        result_entry = {
            "q_id": q_id,
            "evaluation": evaluation,
            "evaluation_result": evaluation_result,
            "question": question,
            "context": context,
            "ground_truth": ground_truth,
            "response": response
        }

        # with open('/root/hsin_research/ruledistill-main/data/testset_results.jsonl', 'a') as f:
        with open('/root/hsin_research/ruledistill-main/data/trainset_results_without_rules.jsonl', 'a') as f:
            f.write(json.dumps(result_entry) + '\n')



# nohup python3 -u test_train_set.py > trainset_run.log 2>&1 &