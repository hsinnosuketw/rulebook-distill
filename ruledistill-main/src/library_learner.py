"""
Library Learner Module

Implements the "Sleep" phase of the Wake-Sleep algorithm for neuro-symbolic
program synthesis. Abstracts common program patterns into reusable rules.

Based on DreamCoder research for library learning.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re

from trace_buffer import TraceBuffer, BufferedTrace
from rulebook_utils import (
    create_sketch_rule,
    serialize_rulebook_with_sketches,
    parse_rulebook_with_sketches,
    parse_rulebook
)


class LibraryLearner:
    """
    Abstracts common program patterns into reusable sketches during Sleep phase.
    
    The learner:
    1. Clusters traces by operation pattern
    2. Abstracts concrete values into slot bindings
    3. Generates new sketch-based rules
    4. Compresses/updates the rulebook
    """
    
    def __init__(
        self,
        min_cluster_size: int = 5,
        max_new_rules: int = 5,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize the library learner.
        
        Args:
            min_cluster_size: Minimum traces needed for abstraction
            max_new_rules: Maximum new rules to add per sleep phase
            confidence_threshold: Minimum confidence for new rules
        """
        self.min_cluster_size = min_cluster_size
        self.max_new_rules = max_new_rules
        self.confidence_threshold = confidence_threshold
    
    def cluster_traces(self, traces: List[BufferedTrace]) -> Dict[str, List[BufferedTrace]]:
        """
        Cluster traces by operation pattern.
        
        Args:
            traces: List of successful traces from buffer
            
        Returns:
            Dictionary mapping patterns to trace lists
        """
        clusters = defaultdict(list)
        for trace in traces:
            clusters[trace.operation_pattern].append(trace)
        return dict(clusters)
    
    def _extract_semantic_hint(self, question: str, arg: str) -> str:
        """
        Try to extract semantic meaning for an argument from the question.
        
        Args:
            question: The question text
            arg: The value that was used
            
        Returns:
            Semantic hint like "new_value", "old_value", "total", etc.
        """
        # Common patterns in financial questions
        year_patterns = [
            (r'(\d{4})', "year"),
            (r'(?:current|new|this|later|ending)\s+(?:year|period)', "new_value"),
            (r'(?:previous|prior|old|last|beginning)\s+(?:year|period)', "old_value"),
        ]
        
        value_patterns = [
            (r'(?:total|sum|aggregate)', "total"),
            (r'(?:average|mean)', "average"),
            (r'(?:revenue|sales|income)', "revenue"),
            (r'(?:cost|expense)', "cost"),
            (r'(?:profit|earnings)', "profit"),
            (r'(?:growth|change|increase|decrease)', "change"),
        ]
        
        question_lower = question.lower()
        
        # Try to match value patterns
        for pattern, semantic in value_patterns:
            if re.search(pattern, question_lower):
                return semantic
        
        # Default based on position
        return "value"
    
    def abstract_pattern(
        self, 
        pattern: str, 
        cluster: List[BufferedTrace]
    ) -> Optional[dict]:
        """
        Generate a sketch rule from a cluster of similar programs.
        
        Args:
            pattern: Operation pattern (e.g., "subtract_divide")
            cluster: List of traces with this pattern
            
        Returns:
            Sketch rule dictionary, or None if abstraction fails
        """
        if len(cluster) < self.min_cluster_size:
            return None
        
        # Get representative program structure
        sample_program = cluster[0].program
        
        # Build sketch template from pattern
        operations = pattern.split("_")
        
        # Analyze argument positions across cluster
        arg_positions = []  # [(step_idx, arg_idx, semantic_hints)]
        
        for step_idx, op in enumerate(operations):
            for arg_idx in [1, 2]:  # arg1 and arg2 positions in token list
                token_idx = step_idx * 4 + arg_idx
                if token_idx < len(sample_program):
                    token = sample_program[token_idx]
                    
                    # Skip if it's a step reference
                    if token.startswith("#"):
                        continue
                    
                    # Collect values from all traces
                    values = []
                    for trace in cluster:
                        if token_idx < len(trace.program):
                            values.append(trace.program[token_idx])
                    
                    # Determine semantic hint from questions
                    semantic_hints = [
                        self._extract_semantic_hint(t.question, t.program[token_idx])
                        for t in cluster if token_idx < len(t.program)
                    ]
                    
                    # Most common semantic
                    if semantic_hints:
                        most_common = max(set(semantic_hints), key=semantic_hints.count)
                    else:
                        most_common = f"arg{len(arg_positions)}"
                    
                    arg_positions.append({
                        'step_idx': step_idx,
                        'arg_idx': arg_idx,
                        'semantic': most_common,
                        'slot_id': f"${len(arg_positions)}"
                    })
        
        # Build sketch template
        sketch_parts = []
        for i, op in enumerate(operations):
            args = []
            for arg_idx in [1, 2]:
                # Find if this position is a slot
                slot = next(
                    (a for a in arg_positions 
                     if a['step_idx'] == i and a['arg_idx'] == arg_idx),
                    None
                )
                if slot:
                    args.append(f"{slot['slot_id']}:{slot['semantic']}")
                else:
                    # It's a reference to previous step
                    args.append(f"#{i-1}" if i > 0 else "$0:value")
            
            sketch_parts.append(f"{op}({args[0]}, {args[1]})")
        
        # Join into nested sketch
        if len(sketch_parts) == 1:
            sketch_template = sketch_parts[0]
        else:
            # For now, keep as sequential operations
            sketch_template = ", ".join(sketch_parts)
        
        # Extract trigger from questions (find common phrases)
        question_words = []
        for trace in cluster[:10]:  # Sample first 10
            words = trace.question.lower().split()
            question_words.extend(words)
        
        # Find most common meaningful words
        word_counts = defaultdict(int)
        stop_words = {"the", "a", "an", "is", "was", "what", "how", "of", "in", "for", "to", "and", "or"}
        for word in question_words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] += 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        trigger = " OR ".join([w[0] for w in top_words]) if top_words else pattern.replace("_", " ")
        
        # Create rule
        rule = create_sketch_rule(
            rule_id=f"LEARNED_{pattern.upper()}",
            trigger=trigger,
            sketch_template=sketch_template,
            rule_type="LEARNED_SKETCH",
            source="library_learner"
        )
        
        # Add confidence based on cluster size
        rule['confidence'] = str(min(len(cluster) / 20.0, 1.0))
        
        return rule
    
    def run_sleep_phase(
        self,
        buffer: TraceBuffer,
        current_rulebook: str = ""
    ) -> Tuple[str, List[dict]]:
        """
        Execute the sleep phase: cluster, abstract, update rulebook.
        
        Args:
            buffer: Trace buffer with successful programs
            current_rulebook: Current rulebook XML string
            
        Returns:
            Tuple of (updated rulebook XML, list of new rules added)
        """
        traces = buffer.get_all_traces()
        
        if not traces:
            return current_rulebook, []
        
        # Cluster by pattern
        clusters = self.cluster_traces(traces)
        
        # Find patterns with enough examples
        abstractable = [
            (p, c) for p, c in clusters.items()
            if len(c) >= self.min_cluster_size
        ]
        
        # Sort by cluster size (most evidence first)
        abstractable.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Generate new rules
        new_rules = []
        for pattern, cluster in abstractable[:self.max_new_rules]:
            rule = self.abstract_pattern(pattern, cluster)
            if rule:
                new_rules.append(rule)
        
        if not new_rules:
            return current_rulebook, []
        
        # Parse existing rules
        existing_rules = parse_rulebook(current_rulebook) if current_rulebook else []
        
        # Merge: add new rules, avoiding duplicates by pattern
        existing_patterns = {
            r.get('id', '').replace('LEARNED_', '').lower()
            for r in existing_rules
        }
        
        for rule in new_rules:
            rule_pattern = rule['id'].replace('LEARNED_', '').lower()
            if rule_pattern not in existing_patterns:
                existing_rules.append(rule)
        
        # Serialize with sketch support
        updated_rulebook = serialize_rulebook_with_sketches(existing_rules)
        
        return updated_rulebook, new_rules
    
    def get_abstraction_stats(
        self, 
        buffer: TraceBuffer
    ) -> Dict:
        """
        Get statistics about potential abstractions.
        
        Args:
            buffer: Trace buffer to analyze
            
        Returns:
            Dictionary with abstraction statistics
        """
        traces = buffer.get_all_traces()
        clusters = self.cluster_traces(traces)
        
        abstractable = [
            (p, len(c)) for p, c in clusters.items()
            if len(c) >= self.min_cluster_size
        ]
        
        return {
            'total_traces': len(traces),
            'unique_patterns': len(clusters),
            'abstractable_patterns': len(abstractable),
            'abstractable_details': sorted(abstractable, key=lambda x: x[1], reverse=True),
            'largest_cluster': max(clusters.values(), key=len) if clusters else []
        }


# Test the library learner
if __name__ == "__main__":
    print("Testing Library Learner...")
    print("=" * 60)
    
    # Create a buffer with sample traces
    buffer = TraceBuffer()
    
    # Add multiple traces with same pattern (subtract_divide)
    growth_questions = [
        "What is the growth rate from 100 to 120?",
        "Calculate the percentage change from 200 to 250",
        "What is the year-over-year growth from 500 to 600?",
        "Find the increase rate from 80 to 100",
        "Determine the growth from 150 to 180",
        "What's the percentage growth from 300 to 360?",
    ]
    
    for i, q in enumerate(growth_questions):
        old = 100 + i * 50
        new = old * 1.2
        program = [
            "subtract(", str(new), str(old), ")",
            "divide(", "#0", str(old), ")",
            "EOF"
        ]
        buffer.add_success(q, program, 0.2)
    
    # Add some percentage calculation traces
    pct_questions = [
        "What portion of 500 is 100?",
        "What fraction of 1000 is 200?",
        "Calculate the percentage of 250 out of 1000",
    ]
    
    for i, q in enumerate(pct_questions):
        total = 500 + i * 250
        part = total * 0.2
        program = ["divide(", str(part), str(total), ")", "EOF"]
        buffer.add_success(q, program, 0.2)
    
    print(f"Buffer stats: {buffer.get_stats()}")
    print()
    
    # Initialize learner
    learner = LibraryLearner(min_cluster_size=3)
    
    # Get abstraction stats
    stats = learner.get_abstraction_stats(buffer)
    print(f"Abstraction stats: {stats}")
    print()
    
    # Run sleep phase
    new_rulebook, new_rules = learner.run_sleep_phase(buffer)
    
    print(f"Generated {len(new_rules)} new rules:")
    for rule in new_rules:
        print(f"  - {rule['id']}: {rule['trigger'][:50]}...")
        if rule.get('sketch'):
            print(f"    Sketch: {rule['sketch']['template']}")
    print()
    
    print("Updated rulebook:")
    print(new_rulebook)
    print()
    
    print("=" * 60)
    print("✅ Library Learner tests complete!")
