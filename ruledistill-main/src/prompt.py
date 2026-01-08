SYSTEM_PROMPT = "You are a financial reasoning expert. \
    Provide only the final numerical answer to the question based on the provided context. \
    Do not include any explanation or other text."

SYSTEM_PROMPT_WITH_RULES_TEMPLATE = """
You are a financial reasoning expert. \
Provide only the final numerical answer to the question based on the provided context. Do not include any explanation or other text.

Follow these rules strictly:
{rules}
"""

USER_PROMPT_TEMPLATE = """
You are a financial reasoning expert. 
You'll receive the Question and the Context required to answer the question.
You can respond directly to the question with your answer.

Context: {context}

Question: {question}
Answer:"""

JUDGE_PROMPT_TEMPLATE = """
[System]
You are an impartial judge evaluating the quality of an AI response.

[Question and Ground Truth]
Question: 
{question}

Ground Truth:
{ground_truth}


[AI Response]
{response}

[Task]
Evaluate the AI response based on correctness, clarity, and helpfulness. 
Provide the True/False answer.

Always answer in the following format:
# Evaluation Explanation
Your explanation

# Final Evaluation
True/False
"""