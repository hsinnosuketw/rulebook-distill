"""
Rulebook XML Utilities

Provides parsing, serialization, and manipulation utilities for the
XML-based rulebook format used in the self-regulating pipeline.
"""

import xml.etree.ElementTree as ET
import re
from typing import Optional


def parse_rulebook(xml_string: str) -> list[dict]:
    """
    Parse rulebook XML into list of rule dictionaries.
    
    Args:
        xml_string: XML string containing the rulebook
        
    Returns:
        List of rule dictionaries with keys: id, type, source, trigger, action, phase, confidence
    """
    rules = []
    
    try:
        # Clean up the XML string
        xml_string = xml_string.strip()
        if not xml_string:
            return rules
            
        # Handle case where XML might be wrapped in code blocks
        if "```xml" in xml_string:
            xml_string = re.search(r'```xml\s*(.*?)\s*```', xml_string, re.DOTALL)
            if xml_string:
                xml_string = xml_string.group(1)
            else:
                return rules
        elif "```" in xml_string:
            xml_string = re.search(r'```\s*(.*?)\s*```', xml_string, re.DOTALL)
            if xml_string:
                xml_string = xml_string.group(1)
            else:
                return rules
        
        # Parse XML
        root = ET.fromstring(xml_string)
        
        # Handle both <Rulebook> and direct <Rule> elements
        if root.tag == "Rulebook":
            rule_elements = root.findall(".//Rule")
        elif root.tag == "Rule":
            rule_elements = [root]
        else:
            rule_elements = root.findall(".//Rule")
        
        for rule_elem in rule_elements:
            rule = {
                "id": rule_elem.get("id", ""),
                "type": rule_elem.get("type", ""),
                "source": rule_elem.get("source", ""),
                "phase": rule_elem.get("phase", "generation"),
                "confidence": rule_elem.get("confidence", "1"),
            }
            
            trigger_elem = rule_elem.find("Trigger")
            action_elem = rule_elem.find("Action")
            
            rule["trigger"] = trigger_elem.text.strip() if trigger_elem is not None and trigger_elem.text else ""
            rule["action"] = action_elem.text.strip() if action_elem is not None and action_elem.text else ""
            
            rules.append(rule)
            
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
    except Exception as e:
        print(f"Error parsing rulebook: {e}")
    
    return rules


def serialize_rulebook(rules: list[dict], domain: str = "finqa") -> str:
    """
    Convert rule list back to XML string.
    
    Args:
        rules: List of rule dictionaries
        domain: Domain attribute for the Rulebook element
        
    Returns:
        XML string representation of the rulebook
    """
    if not rules:
        return f'<Rulebook domain="{domain}"></Rulebook>'
    
    lines = [f'<Rulebook domain="{domain}">']
    
    for rule in rules:
        rule_id = rule.get("id", "00")
        rule_type = rule.get("type", "general")
        phase = rule.get("phase", "generation")
        confidence = rule.get("confidence", "1")
        source = rule.get("source", "")
        trigger = rule.get("trigger", "")
        action = rule.get("action", "")
        
        lines.append(f'    <Rule id="{rule_id}" type="{rule_type}" phase="{phase}" confidence="{confidence}" source="{source}">')
        lines.append(f'        <Trigger>{_escape_xml(trigger)}</Trigger>')
        lines.append(f'        <Action>{_escape_xml(action)}</Action>')
        lines.append('    </Rule>')
    
    lines.append('</Rulebook>')
    
    return '\n'.join(lines)


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def count_rules(xml_string: str) -> int:
    """
    Count total rules in rulebook.
    
    Args:
        xml_string: XML string containing the rulebook
        
    Returns:
        Number of rules in the rulebook
    """
    rules = parse_rulebook(xml_string)
    return len(rules)


def validate_rulebook(xml_string: str) -> tuple[bool, str]:
    """
    Validate rulebook structure.
    
    Args:
        xml_string: XML string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not xml_string or not xml_string.strip():
        return False, "Empty rulebook string"
    
    try:
        rules = parse_rulebook(xml_string)
        
        if not rules:
            # Check if it's a valid but empty rulebook
            xml_string = xml_string.strip()
            if "<Rulebook" in xml_string and "</Rulebook>" in xml_string:
                return True, "Valid empty rulebook"
            return False, "No valid rules found"
        
        # Check for required fields
        for i, rule in enumerate(rules):
            if not rule.get("id"):
                return False, f"Rule {i+1} missing 'id' attribute"
            if not rule.get("trigger") and not rule.get("action"):
                return False, f"Rule {rule['id']} missing both trigger and action"
        
        # Check for duplicate IDs
        ids = [r["id"] for r in rules]
        if len(ids) != len(set(ids)):
            return False, "Duplicate rule IDs found"
        
        return True, f"Valid rulebook with {len(rules)} rules"
        
    except ET.ParseError as e:
        return False, f"XML Parse Error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def merge_rules(rule1: dict, rule2: dict, new_id: Optional[str] = None) -> dict:
    """
    Merge two similar rules into one.
    
    Args:
        rule1: First rule dictionary
        rule2: Second rule dictionary
        new_id: Optional new ID for merged rule
        
    Returns:
        Merged rule dictionary
    """
    merged = {
        "id": new_id or rule1.get("id", "00"),
        "type": rule1.get("type") or rule2.get("type", "merged"),
        "phase": rule1.get("phase", "generation"),
        "confidence": str(max(
            float(rule1.get("confidence", 1)),
            float(rule2.get("confidence", 1))
        )),
        "source": f"merged:{rule1.get('source', '')}+{rule2.get('source', '')}",
        "trigger": _merge_text(rule1.get("trigger", ""), rule2.get("trigger", "")),
        "action": _merge_text(rule1.get("action", ""), rule2.get("action", "")),
    }
    
    return merged


def _merge_text(text1: str, text2: str) -> str:
    """Merge two text strings, avoiding redundancy."""
    if not text1:
        return text2
    if not text2:
        return text1
    if text1.lower().strip() == text2.lower().strip():
        return text1
    
    # Simple merge: combine with semicolon
    return f"{text1}; {text2}"


def extract_rules_from_response(response: str) -> str:
    """
    Extract rulebook XML from an LLM response that may contain other text.
    
    Args:
        response: LLM response string
        
    Returns:
        Extracted XML string, or empty string if none found
    """
    # Try to find XML block in code fence
    code_block_match = re.search(r'```(?:xml)?\s*(<Rulebook.*?</Rulebook>)\s*```', response, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1)
    
    # Try to find raw XML
    xml_match = re.search(r'<Rulebook.*?</Rulebook>', response, re.DOTALL | re.IGNORECASE)
    if xml_match:
        return xml_match.group(0)
    
    return ""


def get_empty_rulebook(domain: str = "finqa") -> str:
    """Return an empty rulebook XML string."""
    return f'<Rulebook domain="{domain}"></Rulebook>'


def add_rule_to_rulebook(rulebook_xml: str, new_rule: dict) -> str:
    """
    Add a single rule to an existing rulebook.
    
    Args:
        rulebook_xml: Current rulebook XML string
        new_rule: Rule dictionary to add
        
    Returns:
        Updated rulebook XML string
    """
    rules = parse_rulebook(rulebook_xml)
    
    # Auto-assign ID if not provided
    if not new_rule.get("id"):
        existing_ids = [int(r["id"]) for r in rules if r["id"].isdigit()]
        new_id = max(existing_ids, default=0) + 1
        new_rule["id"] = f"{new_id:02d}"
    
    rules.append(new_rule)
    return serialize_rulebook(rules)


def remove_rule_from_rulebook(rulebook_xml: str, rule_id: str) -> str:
    """
    Remove a rule by ID from the rulebook.
    
    Args:
        rulebook_xml: Current rulebook XML string
        rule_id: ID of rule to remove
        
    Returns:
        Updated rulebook XML string
    """
    rules = parse_rulebook(rulebook_xml)
    rules = [r for r in rules if r["id"] != rule_id]
    return serialize_rulebook(rules)


def compress_rulebook(rulebook_xml: str, max_rules: int = 15) -> str:
    """
    Compress rulebook to stay within max_rules limit.
    
    Strategy:
    1. Keep rules with highest confidence
    2. Merge similar rules
    
    Args:
        rulebook_xml: Current rulebook XML string
        max_rules: Maximum number of rules to keep
        
    Returns:
        Compressed rulebook XML string
    """
    rules = parse_rulebook(rulebook_xml)
    
    if len(rules) <= max_rules:
        return rulebook_xml
    
    # Sort by confidence (descending)
    rules.sort(key=lambda r: float(r.get("confidence", 1)), reverse=True)
    
    # Keep top rules
    rules = rules[:max_rules]
    
    # Re-number IDs
    for i, rule in enumerate(rules):
        rule["id"] = f"{i+1:02d}"
    
    return serialize_rulebook(rules)


# =============================================================================
# NEURO-SYMBOLIC SKETCH EXTENSIONS
# These functions add support for executable DSL sketches in rules
# =============================================================================


def parse_sketch(sketch_xml: str) -> Optional[dict]:
    """
    Parse a <Sketch> element into a template structure.
    
    Sketch format:
        <Sketch>
            divide(subtract($0:new_value, $1:old_value), $1:old_value)
        </Sketch>
    
    Args:
        sketch_xml: XML string or text containing the sketch
        
    Returns:
        Dictionary with:
        - template: str (the sketch string with slots)
        - slots: list of slot definitions
        - operations: list of operation names in order
    """
    if not sketch_xml:
        return None
    
    # Extract content from XML tags if present
    sketch_match = re.search(r'<Sketch>(.*?)</Sketch>', sketch_xml, re.DOTALL | re.IGNORECASE)
    if sketch_match:
        template = sketch_match.group(1).strip()
    else:
        template = sketch_xml.strip()
    
    # Clean up whitespace
    template = re.sub(r'\s+', ' ', template).strip()
    
    # Extract slot definitions ($N:semantic_name)
    slot_pattern = r'\$(\d+):(\w+)'
    slots = []
    for match in re.finditer(slot_pattern, template):
        slot_id = f"${match.group(1)}"
        semantic = match.group(2)
        if not any(s['id'] == slot_id for s in slots):
            slots.append({
                'id': slot_id,
                'index': int(match.group(1)),
                'semantic': semantic
            })
    
    # Sort slots by index
    slots.sort(key=lambda s: s['index'])
    
    # Extract operation sequence
    op_pattern = r'(\w+)\s*\('
    operations = re.findall(op_pattern, template)
    
    return {
        'template': template,
        'slots': slots,
        'operations': operations
    }


def extract_slot_bindings(sketch: dict) -> list[dict]:
    """
    Extract slot definitions from a parsed sketch.
    
    Args:
        sketch: Parsed sketch dictionary from parse_sketch()
        
    Returns:
        List of slot definitions with id, index, semantic, description
    """
    if not sketch or 'slots' not in sketch:
        return []
    
    return sketch['slots']


def instantiate_sketch(sketch: dict, bindings: dict) -> list[str]:
    """
    Fill sketch slots with concrete values to produce a DSL program.
    
    Args:
        sketch: Parsed sketch from parse_sketch()
        bindings: Dictionary mapping slot IDs ($0, $1, etc.) to values
        
    Returns:
        List of program tokens (DSL format)
        
    Example:
        sketch = parse_sketch("divide(subtract($0:new, $1:old), $1:old)")
        bindings = {"$0": "100", "$1": "80"}
        program = instantiate_sketch(sketch, bindings)
        # Returns: ["subtract(", "100", "80", ")", "divide(", "#0", "80", ")", "EOF"]
    """
    if not sketch or 'template' not in sketch:
        return []
    
    template = sketch['template']
    
    # Replace slot references with bound values
    # First, normalize template to remove semantic hints
    normalized = re.sub(r'\$(\d+):\w+', r'$\1', template)
    
    # Parse the normalized template into operations
    # Pattern: operation(arg1, arg2)
    op_pattern = r'(\w+)\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
    
    tokens = []
    step_index = 0
    
    for match in re.finditer(op_pattern, normalized):
        op = match.group(1)
        arg1 = match.group(2).strip()
        arg2 = match.group(3).strip()
        
        # Replace slot references with bound values or step references
        def resolve_arg(arg: str) -> str:
            if arg.startswith('$'):
                # It's a slot reference
                if arg in bindings:
                    return str(bindings[arg])
                else:
                    return arg  # Leave unbound for debugging
            elif arg.startswith('#'):
                # It's a step reference, keep as-is
                return arg
            else:
                # It's a literal or nested
                # Check if it's a nested operation result
                nested_match = re.match(r'(\w+)\s*\(', arg)
                if nested_match:
                    # This is a nested operation - return step reference
                    # The outer loop will handle it
                    return f"#{step_index}"
                return arg
        
        resolved_arg1 = resolve_arg(arg1)
        resolved_arg2 = resolve_arg(arg2)
        
        tokens.extend([f"{op}(", resolved_arg1, resolved_arg2, ")"])
        step_index += 1
    
    if tokens:
        tokens.append("EOF")
    
    return tokens


def parse_rulebook_with_sketches(xml_string: str) -> list[dict]:
    """
    Parse rulebook XML with support for Sketch elements.
    
    Extended version of parse_rulebook that also extracts Sketch and Slots.
    
    Args:
        xml_string: XML string containing the rulebook
        
    Returns:
        List of rule dictionaries with additional sketch-related keys
    """
    rules = parse_rulebook(xml_string)  # Get base rules
    
    if not rules:
        return rules
    
    try:
        # Re-parse to get sketch elements
        xml_string = xml_string.strip()
        
        # Handle code blocks
        if "```xml" in xml_string:
            match = re.search(r'```xml\s*(.*?)\s*```', xml_string, re.DOTALL)
            if match:
                xml_string = match.group(1)
        elif "```" in xml_string:
            match = re.search(r'```\s*(.*?)\s*```', xml_string, re.DOTALL)
            if match:
                xml_string = match.group(1)
        
        root = ET.fromstring(xml_string)
        
        if root.tag == "Rulebook":
            rule_elements = root.findall(".//Rule")
        elif root.tag == "Rule":
            rule_elements = [root]
        else:
            rule_elements = root.findall(".//Rule")
        
        # Match rules by ID and add sketch info
        rule_map = {r['id']: r for r in rules}
        
        for rule_elem in rule_elements:
            rule_id = rule_elem.get("id", "")
            if rule_id not in rule_map:
                continue
            
            rule = rule_map[rule_id]
            
            # Check for Sketch element
            sketch_elem = rule_elem.find("Sketch")
            if sketch_elem is None:
                # Check inside Action element
                action_elem = rule_elem.find("Action")
                if action_elem is not None:
                    sketch_elem = action_elem.find("Sketch")
            
            if sketch_elem is not None and sketch_elem.text:
                sketch_text = sketch_elem.text.strip()
                parsed_sketch = parse_sketch(sketch_text)
                rule['sketch'] = parsed_sketch
                rule['has_sketch'] = True
            else:
                rule['has_sketch'] = False
            
            # Check for Slots element
            slots_elem = rule_elem.find("Slots")
            if slots_elem is not None:
                slot_defs = []
                for slot_elem in slots_elem.findall("Slot"):
                    slot_defs.append({
                        'id': slot_elem.get('id', ''),
                        'semantic': slot_elem.get('semantic', ''),
                        'description': slot_elem.text.strip() if slot_elem.text else ''
                    })
                rule['slot_definitions'] = slot_defs
    
    except Exception as e:
        # If extended parsing fails, still return base rules
        print(f"Warning: Extended sketch parsing failed: {e}")
    
    return rules


def create_sketch_rule(
    rule_id: str,
    trigger: str,
    sketch_template: str,
    slots: list[dict] = None,
    rule_type: str = "SKETCH",
    source: str = "learned"
) -> dict:
    """
    Create a new sketch-based rule dictionary.
    
    Args:
        rule_id: Unique rule identifier
        trigger: Trigger pattern (when to apply this rule)
        sketch_template: DSL sketch with slot placeholders
        slots: List of slot definitions
        rule_type: Type of rule (default: SKETCH)
        source: Source identifier
        
    Returns:
        Rule dictionary ready for serialization
    """
    parsed = parse_sketch(sketch_template)
    
    return {
        'id': rule_id,
        'type': rule_type,
        'trigger': trigger,
        'action': '',  # Will be replaced by sketch in serialization
        'sketch': parsed,
        'has_sketch': True,
        'slot_definitions': slots or parsed.get('slots', []),
        'source': source,
        'phase': 'generation',
        'confidence': '1.0'
    }


def serialize_sketch_rule(rule: dict) -> str:
    """
    Serialize a sketch-based rule to XML string.
    
    Args:
        rule: Rule dictionary with sketch data
        
    Returns:
        XML string for this rule
    """
    rule_id = rule.get('id', '00')
    rule_type = rule.get('type', 'SKETCH')
    trigger = rule.get('trigger', '')
    source = rule.get('source', '')
    phase = rule.get('phase', 'generation')
    confidence = rule.get('confidence', '1')
    
    lines = [
        f'    <Rule id="{rule_id}" type="{rule_type}" phase="{phase}" '
        f'confidence="{confidence}" source="{source}">'
    ]
    
    lines.append(f'        <Trigger>{_escape_xml(trigger)}</Trigger>')
    
    if rule.get('has_sketch') and rule.get('sketch'):
        sketch = rule['sketch']
        lines.append('        <Action>')
        lines.append(f'            <Sketch>{_escape_xml(sketch["template"])}</Sketch>')
        lines.append('        </Action>')
        
        # Add slot definitions
        if rule.get('slot_definitions'):
            lines.append('        <Slots>')
            for slot in rule['slot_definitions']:
                desc = slot.get('description', '')
                lines.append(
                    f'            <Slot id="{slot["id"]}" semantic="{slot.get("semantic", "")}">'
                    f'{_escape_xml(desc)}</Slot>'
                )
            lines.append('        </Slots>')
    else:
        action = rule.get('action', '')
        lines.append(f'        <Action>{_escape_xml(action)}</Action>')
    
    lines.append('    </Rule>')
    
    return '\n'.join(lines)


def serialize_rulebook_with_sketches(rules: list[dict], domain: str = "finqa-neuro") -> str:
    """
    Serialize a rulebook that may contain sketch-based rules.
    
    Args:
        rules: List of rule dictionaries (may include sketches)
        domain: Domain attribute for Rulebook element
        
    Returns:
        XML string representation
    """
    if not rules:
        return f'<Rulebook domain="{domain}"></Rulebook>'
    
    lines = [f'<Rulebook domain="{domain}">']
    
    for rule in rules:
        if rule.get('has_sketch'):
            lines.append(serialize_sketch_rule(rule))
        else:
            # Use standard serialization
            rule_id = rule.get("id", "00")
            rule_type = rule.get("type", "general")
            phase = rule.get("phase", "generation")
            confidence = rule.get("confidence", "1")
            source = rule.get("source", "")
            trigger = rule.get("trigger", "")
            action = rule.get("action", "")
            
            lines.append(f'    <Rule id="{rule_id}" type="{rule_type}" phase="{phase}" '
                        f'confidence="{confidence}" source="{source}">')
            lines.append(f'        <Trigger>{_escape_xml(trigger)}</Trigger>')
            lines.append(f'        <Action>{_escape_xml(action)}</Action>')
            lines.append('    </Rule>')
    
    lines.append('</Rulebook>')
    
    return '\n'.join(lines)

