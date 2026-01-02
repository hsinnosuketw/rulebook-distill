rulebook_xml_content = """
<Rulebook domain="finqa_reasoning">
    <Rule id="01" phase="generation" confidence="1" source="results.jsonl">
        <Trigger>questions involves net changes, decreases, or negative balances</Trigger>
        <Action>CRITICAL FORMATTING RULE: Always preserve the negative sign (-) for values indicating a decrease, net loss, or negative balance. Do not output absolute values unless specifically asked for "magnitude".</Action>
    </Rule>
    <Rule id="02" phase="generation" confidence="1" source="results.jsonl">
        <Trigger>questions asking for percentage rates or growth rates</Trigger>
        <Action>CRITICAL FORMATTING RULE: Provide percentage values with at least two decimal places of precision (e.g., 1.33% instead of 1.3%) unless the value is an exact integer. Ensure the decimal point is correctly placed relative to the % sign.</Action>
    </Rule>
    <Rule id="03" phase="generation" confidence="1" source="results.jsonl">
        <Trigger>questions specifying units like "in millions", "in billions", or "in thousands"</Trigger>
        <Action>KNOWLEDGE INJECTION: Verify the unit scale of the numbers in the context. If the question asks for "in millions" and the context is in millions, do not multiply by 1,000,000; output the number as it corresponds to the requested unit.</Action>
    </Rule>
    <Rule id="04" phase="generation" confidence="1" source="results.jsonl">
        <Trigger>numerical calculation following Multi-step arithmetic</Trigger>
        <Action>CRITICAL FORMATTING RULE: Output ONLY the final numerical result. Never include the formula, scratchpad, or intermediate steps like "(A + B) / C = D" in the final message content.</Action>
    </Rule>
    <Rule id="05" phase="generation" confidence="1" source="results.jsonl">
        <Trigger>questions about specific years or line items appearing in tables</Trigger>
        <Action>KNOWLEDGE INJECTION: Perform a granular search of all provided context snippets and tables before concluding that data is missing. Look for headers, footnotes, and parenthetical info that may contain the target year or item.</Action>
    </Rule>
    <Rule id="06" phase="generation" confidence="1" source="results.jsonl">
        <Trigger>questions involving ratios or comparisons</Trigger>
        <Action>CRITICAL FORMATTING RULE: When calculating a ratio, ensure the numerator and denominator are correctly ordered as per the question phrasing (e.g., "ratio of A to B" is A/B).</Action>
    </Rule>
</Rulebook>
"""
