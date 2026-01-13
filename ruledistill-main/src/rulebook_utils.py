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
