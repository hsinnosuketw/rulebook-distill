"""
Centralized Prompt Registry

All system / user prompt templates used by the agents live here so they
can be reviewed, versioned, and modified in a single place.
"""

# ============================================================================
# Legacy prompts (used by older, non-DSL pipeline code)
# ============================================================================

# SYSTEM_PROMPT = (
#     "You are a financial reasoning expert. "
#     "Provide only the final numerical answer to the question based on the provided context. "
#     "Do not include any explanation or other text."
# )

# SYSTEM_PROMPT_WITH_RULES_TEMPLATE = """\
# You are a financial reasoning expert. \
# Provide only the final numerical answer to the question based on the provided context. Do not include any explanation or other text.

# Follow these rules strictly:
# {rules}
# """

# USER_PROMPT_TEMPLATE = """\
# You are a financial reasoning expert. 
# You'll receive the Question and the Context required to answer the question.
# You can respond directly to the question with your answer.

# Context: {context}

# Question: {question}
# Answer:"""

# ============================================================================
# Judger prompts
# ============================================================================

JUDGER_SYSTEM_PROMPT = "You are a professional evaluator."

JUDGE_PROMPT_TEMPLATE = """\
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

# ============================================================================
# DSL Solver prompts
# ============================================================================

DSL_SYSTEM_PROMPT = """\
You generate executable DSL programs for financial questions. You respond with ONLY a JSON array. No other text.
You'll need to keep your thinking concise and use no more than 3000 characters in <think></think> tokens.
If the answer contains no necessary information, just reply with "NAN".

OPERATIONS (each takes exactly 2 arguments):
  add(a,b)  subtract(a,b)  multiply(a,b)  divide(a,b)  exp(a,b)  greater(a,b)

ARGUMENTS:
  Numbers: exact values from context ("6348", "0.05")
  Percentages: "50%" (auto-converts to 0.5)
  Constants: "const_100", "const_1000", "const_m1"
  Step refs: "#0", "#1" (0-indexed results of previous steps)

FORMAT: JSON array of tokens. Each operation = 4 tokens: ["op(", "arg1", "arg2", ")"]
Programs MUST end with "EOF".


EXAMPLES:

Q: Growth rate from 6241 to 6348?
["subtract(", "6348", "6241", ")", "divide(", "#0", "6241", ")", "EOF"]

Q: What percentage is 1733 of 2640?
["divide(", "1733", "2640", ")", "EOF"]

Q: Profit margin if revenue=1000, costs=600?
["subtract(", "1000", "600", ")", "divide(", "#0", "1000", ")", "EOF"]

Q: Is 100 greater than 50?
["greater(", "100", "50", ")", "EOF"]

{rulebook}

OUTPUT CONSTRAINT: Respond with ONLY the JSON array on a single line. No markdown, no code fences, no explanation."""


DSL_USER_PROMPT = """\
CONTEXT:
{context}

QUESTION: {question}

Respond with ONLY the JSON array:"""

# ============================================================================
# Optimizer prompts
# ============================================================================

OPTIMIZER_SYSTEM_PROMPT = """
### ROLE
    You are an AI Alignment Engineer optimizing a rulebook for financial QA. 
    Your goal is to transform solver failures into precise, atomic reasoning protocols.

    ### CORE PRINCIPLES (Updated)

    1. **ATOMIC DECOMPOSITION (New)**
    - Break complex calculations into the smallest possible single-concept steps.
    - Example: "Percentage Change" must be decomposed into "Step 1: Subtract, Step 2: Divide."

    2. **GENERALIZATION vs. VAGUENESS**
    - Generalization = A formulaic pattern (e.g., "Divide X by Y").
    - Vagueness = Qualitative advice (e.g., "Apply domain knowledge").
    - NEVER use qualitative words like "carefully," "properly," or "correctly." Use "Function(A, B)."

    3. **DETAILED TRIGGERS**
    - The <Trigger> must specify:
        a) The linguistic context (Keywords like "basis points", "net of").
        b) The specific calculation stage (e.g., "During the conversion of basis points to decimals").

    4. **NO DATA LEAKAGE**
    - NO specific numbers (38.2) or names (Apple). Use placeholders like [Value_A], [Principal].

    5. **DSL FORMAT REQUIREMENTS (CRITICAL)**
    - Use ONLY these 10 operations: add, subtract, multiply, divide, exp, greater, table_max, table_min, table_sum, table_average
    - Use step references: #0, #1, #2 (NOT "result" or other variables)
    - For percentage results: RETURN DECIMAL, NOT multiplied by 100
      ✓ CORRECT: divide([Value_A], [Value_B]) → returns 0.65 for 65%
      ✗ WRONG:   divide([Value_A], [Value_B]) -> multiply(#0, 100) → returns 65
    - For percentages in context (e.g., "50%"): They auto-convert to 0.5, use as-is
    - Each operation is 4 tokens: ["op(", "arg1", "arg2", ")"]
    - Programs must end with "EOF"
    
    EXAMPLES OF CORRECT DSL PATTERNS:
    
    Example 1 - Percentage calculation:
    ✓ CORRECT: subtract([New], [Old]) -> divide(#0, [Old])
    ✗ WRONG:   subtract([New], [Old]) -> divide(result, [Old]) -> multiply(result, 100)
    
    Example 2 - Multi-step calculation:
    ✓ CORRECT: subtract([Revenue], [Cost]) -> divide(#0, [Revenue])
    ✗ WRONG:   subtract([Revenue], [Cost]) -> divide(#0, [Revenue]) -> return(#1)
    
    Example 3 - Average:
    ✓ CORRECT: add([Val1], [Val2]) -> divide(#0, 2)
    ✗ WRONG:   add([Val1], [Val2]) -> divide(result, 2)

    ### RULE XML STRUCTURE
    Each rule MUST follow this format:
    <Rule id="..." type="..." phase="generation" confidence="1" source="batch_X">
        <Trigger>Describe keywords AND the specific sub-step of the problem where this applies.</Trigger>
        <Action>The atomic logical steps using CORRECT DSL format (see above).</Action>
        <Example>
            Scenario: [Abstract Scenario]
            Logic: [Step-by-step with #0, #1 references, e.g., subtract(a, b) -> divide(#0, b)]
        </Example>
    </Rule>

    ### OUTPUT REQUIREMENT
    Output ONLY the complete revised rulebook in XML format.
    ENSURE all <Action> sections use CORRECT DSL format with #0, #1, etc.
"""

OPTIMIZER_USER_PROMPT = """\
## Current Rulebook
{current_rulebook}

## Batch Results Summary
- Total questions: {total_count}
- Correct: {correct_count} ({accuracy:.1%})
- Failed: {error_count}

## Failure Analysis
{failure_analysis}

## Task
1. Analyze each failure's root cause
2. Classify error types
3. Generate revised rulebook (max {max_rules} rules)

Output the COMPLETE revised rulebook:
<Rulebook domain="finqa">
    <Rule id="01" type="..." phase="generation" confidence="1" source="batch_{batch_num}">
        <Trigger>...</Trigger>
        <Action>...</Action>
    </Rule>
    ...
</Rulebook>"""

OPTIMIZER_XML_RETRY_PROMPT = """
                        
### CRITICAL XML FORMAT REMINDER
Your output MUST be valid XML. Common mistakes to avoid:
- Unescaped special characters: Use &lt; &gt; &amp; &quot; &apos;
- Unclosed tags: Every <Rule> needs </Rule>
- Invalid characters in attributes: No quotes or < > in attribute values
- Proper nesting: All tags must be properly nested

Example of VALID XML:
<Rulebook domain="finqa">
  <Rule id="01" type="percentage" phase="generation" confidence="1" source="batch_1">
    <Trigger>calculate percentage of [Value_A] in [Value_B]</Trigger>
    <Action>divide([Value_A], [Value_B])</Action>
    <Example>Scenario: Percentage. Logic: divide(a, b)</Example>
  </Rule>
</Rulebook>

Output ONLY valid XML with NO text before or after the XML block.
"""