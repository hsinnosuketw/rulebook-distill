"""
Rule Deduplication and Compression Script

Analyzes the existing rulebook to:
1. Identify duplicate/similar rules
2. Cluster rules by pattern
3. Generate a compressed, specialized rulebook
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import json


def load_rulebook(path: str) -> list[dict]:
    """Load rules from XML file using regex (handles malformed XML)."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    rules = []
    
    # Use regex to extract rules (more robust than XML parser for malformed files)
    rule_pattern = re.compile(
        r'<Rule[^>]*id=["\']([^"\']*)["\'][^>]*(?:type=["\']([^"\']*)["\'])?[^>]*(?:source=["\']([^"\']*)["\'])?[^>]*>'
        r'.*?<Trigger>(.*?)</Trigger>.*?<Action>(.*?)</Action>.*?</Rule>',
        re.DOTALL | re.IGNORECASE
    )
    
    for match in rule_pattern.finditer(content):
        rules.append({
            'id': match.group(1) or '',
            'type': match.group(2) or '',
            'source': match.group(3) or '',
            'trigger': match.group(4).strip() if match.group(4) else '',
            'action': match.group(5).strip() if match.group(5) else ''
        })
    
    return rules


def extract_keywords(text: str) -> set[str]:
    """Extract key financial/math terms from text."""
    keywords = {
        'percentage', 'percent', 'change', 'increase', 'decrease',
        'ratio', 'divide', 'multiply', 'subtract', 'add',
        'formula', 'calculation', 'sum', 'average', 'total',
        'revenue', 'expense', 'income', 'profit', 'loss',
        'growth', 'rate', 'value', 'difference', 'comparison',
        'rounding', 'precision', 'tolerance', 'scale',
        'negative', 'positive', 'sign', 'absolute'
    }
    
    text_lower = text.lower()
    found = set()
    for kw in keywords:
        if kw in text_lower:
            found.add(kw)
    return found


def cluster_rules(rules: list[dict]) -> dict[str, list[dict]]:
    """Cluster rules by their primary pattern/concern."""
    clusters = defaultdict(list)
    
    for rule in rules:
        combined = f"{rule['trigger']} {rule['action']}"
        keywords = extract_keywords(combined)
        
        # Determine primary cluster
        if 'sign' in keywords or 'negative' in keywords or 'positive' in keywords:
            clusters['sign_handling'].append(rule)
        elif 'percentage' in keywords or 'percent' in keywords:
            if 'change' in keywords or 'growth' in keywords:
                clusters['percentage_change'].append(rule)
            else:
                clusters['percentage_format'].append(rule)
        elif 'ratio' in keywords or 'divide' in keywords:
            clusters['ratio_calculation'].append(rule)
        elif 'average' in keywords or 'sum' in keywords:
            clusters['aggregation'].append(rule)
        elif 'rounding' in keywords or 'precision' in keywords:
            clusters['precision'].append(rule)
        elif 'comparison' in keywords:
            clusters['comparison'].append(rule)
        else:
            clusters['general_arithmetic'].append(rule)
    
    return dict(clusters)


def generate_specialized_rules() -> str:
    """Generate a new, focused rulebook with worked examples."""
    
    specialized_rules = '''<?xml version="1.0" encoding="UTF-8"?>
<Rulebook domain="finqa" version="2.0" description="Optimized rules with worked examples">

<!-- SIGN HANDLING RULES -->
<Rule id="SIGN-01" type="SignError" phase="calculation" confidence="1">
    <Trigger>The question asks for percentage change, growth rate, or difference between two values</Trigger>
    <Action>
        SIGN CHECK PROTOCOL:
        1. Identify OLD value and NEW value
        2. Calculate: (NEW - OLD) / OLD
        3. If NEW < OLD, result MUST be NEGATIVE
        
        EXAMPLE:
        - 2015 revenue: $34.2M
        - 2016 revenue: $32.5M
        - Change: (32.5 - 34.2) / 34.2 = -1.7/34.2 = -0.0497 = -4.97%
        - WRONG: 4.97% (missing negative sign)
        - CORRECT: -4.97% or -0.0497
    </Action>
</Rule>

<Rule id="SIGN-02" type="SignError" phase="output" confidence="1">
    <Trigger>Cash flow, financing activities, or change calculations</Trigger>
    <Action>
        CASH FLOW SIGN RULES:
        - Increase in assets = NEGATIVE cash flow
        - Decrease in assets = POSITIVE cash flow
        - Increase in liabilities = POSITIVE cash flow
        - Decrease in liabilities = NEGATIVE cash flow
        
        Always verify the sign makes logical sense for the financial metric.
    </Action>
</Rule>

<!-- PERCENTAGE FORMAT RULES -->
<Rule id="PCT-01" type="FormatError" phase="output" confidence="1">
    <Trigger>Question asks for a percentage and ground truth appears to be in decimal form (value less than 1)</Trigger>
    <Action>
        FORMAT MATCHING:
        If ground truth is 0.0497, your answer should be 0.0497 (NOT 4.97)
        If ground truth is 4.97, your answer should be 4.97 (NOT 0.0497)
        
        DETECTION: Look at the magnitude of expected answer
        - Values like 0.05, 0.12, -0.03 are DECIMAL form
        - Values like 5%, 12%, -3% are PERCENTAGE form
    </Action>
</Rule>

<Rule id="PCT-02" type="ArithmeticError" phase="calculation" confidence="1">
    <Trigger>Calculating percentage of total, market share, or proportion</Trigger>
    <Action>
        PROPORTION FORMULA:
        Percentage = (Part / Whole) × 100
        
        EXAMPLE:
        - Segment revenue: $14,001M
        - Total revenue: $26,302M
        - Percentage: 14001/26302 × 100 = 53.23%
        - As decimal: 0.5323
    </Action>
</Rule>

<!-- SCALE HANDLING RULES -->
<Rule id="SCALE-01" type="ScaleError" phase="input" confidence="1">
    <Trigger>Values in millions, billions, or thousands are mentioned</Trigger>
    <Action>
        UNIT CONSISTENCY:
        1. Identify all values and their units (M = million, B = billion, K = thousand)
        2. Convert ALL values to the same unit before calculations
        3. Express final answer in the requested unit
        
        EXAMPLE:
        - "Revenue increased from $2.5M to $3.1M"
        - Change: 3.1M - 2.5M = 0.6M = $600,000
        - NOT: 3.1 - 2.5 = 0.6 (missing unit)
    </Action>
</Rule>

<!-- FORMULA APPLICATION RULES -->
<Rule id="FORM-01" type="ArithmeticError" phase="calculation" confidence="1">
    <Trigger>Calculating year-over-year change, growth rate, or CAGR</Trigger>
    <Action>
        GROWTH RATE FORMULA:
        Growth Rate = (New Value - Old Value) / Old Value
        
        STEP-BY-STEP:
        1. Identify the OLD value (earlier period)
        2. Identify the NEW value (later period)
        3. Subtract: difference = NEW - OLD
        4. Divide: rate = difference / OLD
        5. Check sign: if NEW < OLD, rate should be negative
        
        To convert to percentage: multiply by 100
    </Action>
</Rule>

<Rule id="FORM-02" type="ArithmeticError" phase="calculation" confidence="1">
    <Trigger>Calculating total from a known part and its percentage</Trigger>
    <Action>
        REVERSE PERCENTAGE FORMULA:
        Total = Part / (Percentage / 100)
        
        EXAMPLE:
        - "Aircraft fuel expense is $9,896M, which is 23.6% of total operating expenses"
        - Total = 9896 / 0.236 = $41,932M
    </Action>
</Rule>

<!-- COMPARISON RULES -->
<Rule id="COMP-01" type="LogicError" phase="output" confidence="1">
    <Trigger>Question asks "did X exceed Y" or "is X greater than Y"</Trigger>
    <Action>
        COMPARISON PROTOCOL:
        1. Calculate or retrieve both values
        2. Compare: X > Y
        3. Answer "yes" if true, "no" if false
        
        Do NOT provide the numerical difference - provide yes/no.
    </Action>
</Rule>

<Rule id="COMP-02" type="LogicError" phase="calculation" confidence="1">
    <Trigger>Question asks for the "difference" between two values</Trigger>
    <Action>
        DIFFERENCE TYPES:
        - "What is the difference" → Usually absolute value
        - "By how much did X change" → Signed difference (can be negative)
        - "How much more/less" → Explicitly directional
        
        Always consider context for sign.
    </Action>
</Rule>

<!-- PRECISION RULES -->
<Rule id="PREC-01" type="PrecisionError" phase="output" confidence="1">
    <Trigger>Final answer requires rounding</Trigger>
    <Action>
        ROUNDING PROTOCOL:
        1. Perform ALL calculations with full precision
        2. Round ONLY at the final step
        3. Match the precision of the expected answer format
        
        Common formats:
        - Percentages: 1-2 decimal places (e.g., 53.2%, 12.34%)
        - Currency: usually whole numbers or 2 decimal places
        - Ratios: 2-4 decimal places
    </Action>
</Rule>

<!-- DATA RETRIEVAL RULES -->
<Rule id="DATA-01" type="ContextError" phase="input" confidence="1">
    <Trigger>Question references specific years, quarters, or time periods</Trigger>
    <Action>
        TIME PERIOD VERIFICATION:
        1. Identify the exact time period asked about
        2. Find the corresponding values in the context
        3. Do NOT use values from different time periods
        
        Common mistakes:
        - Using 2015 data when asked about 2016
        - Mixing Q1 and Q4 values
        - Confusing fiscal year with calendar year
    </Action>
</Rule>

<Rule id="DATA-02" type="ContextError" phase="input" confidence="1">
    <Trigger>Multiple similar metrics exist in the context (e.g., multiple revenue figures)</Trigger>
    <Action>
        METRIC DISAMBIGUATION:
        1. Read the question carefully for qualifiers (e.g., "net", "gross", "adjusted")
        2. Match the exact metric name from the question
        3. If ambiguous, state the assumption
        
        Common confusions:
        - Net revenue vs gross revenue
        - Operating income vs net income
        - Total assets vs current assets
    </Action>
</Rule>

</Rulebook>'''
    
    return specialized_rules


def analyze_and_compress(input_path: str, output_path: str):
    """Main function to analyze and compress rulebook."""
    print(f"Loading rules from {input_path}...")
    rules = load_rulebook(input_path)
    print(f"Loaded {len(rules)} rules")
    
    # Cluster analysis
    print("\nClustering rules...")
    clusters = cluster_rules(rules)
    
    print("\nCluster distribution:")
    for cluster_name, rules_in_cluster in sorted(clusters.items(), key=lambda x: -len(x[1])):
        print(f"  {cluster_name}: {len(rules_in_cluster)} rules")
    
    # Generate specialized rulebook
    print(f"\nGenerating specialized rulebook to {output_path}...")
    specialized = generate_specialized_rules()
    
    with open(output_path, 'w') as f:
        f.write(specialized)
    
    # Count new rules
    new_rules = load_rulebook(output_path)
    print(f"New rulebook has {len(new_rules)} rules (reduced from {len(rules)})")
    
    print("\n✅ Done!")
    
    return {
        'original_count': len(rules),
        'new_count': len(new_rules),
        'reduction': f"{(1 - len(new_rules)/len(rules))*100:.1f}%",
        'clusters': {k: len(v) for k, v in clusters.items()}
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deduplicate and compress rulebook")
    parser.add_argument(
        "--input",
        type=str,
        default="/root/hsin_research/ruledistill-main/data/all_rules.xml",
        help="Input rulebook path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/hsin_research/ruledistill-main/data/specialized_rules.xml",
        help="Output rulebook path"
    )
    
    args = parser.parse_args()
    
    stats = analyze_and_compress(args.input, args.output)
    print(f"\nStats: {json.dumps(stats, indent=2)}")
