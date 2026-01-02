SYSTEM_PROMPT = "You are a financial reasoning expert. \
    Provide only the final numerical answer to the question based on the provided context. \
    Do not include any explanation or other text."

SYSTEM_PROMPT_WITH_RULES_TEMPLATE = """
You are a financial reasoning expert. \
Provide only the final numerical answer to the question based on the provided context. Do not include any explanation or other text.

Follow these rules strictly:
{rules}
"""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Numerical Answer:"""