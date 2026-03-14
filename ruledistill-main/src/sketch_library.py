"""
Sketch Library Module

Maintains a persistent library of learned DSL sketches that are fed to the solver.
Unlike the NL rulebook which gets overwritten each batch, the sketch library
accumulates and persists across batches.

This is the key integration point for neuro-symbolic program induction.
"""

from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime


class SketchLibrary:
    """
    Persistent library of learned DSL sketches.
    
    The library:
    1. Stores sketch rules from sleep phases
    2. Persists across batches (not overwritten by optimizer)
    3. Generates solver-ready prompt sections
    """
    
    def __init__(self, checkpoint_path: str = None, max_sketches: int = 10):
        """
        Initialize the sketch library.
        
        Args:
            checkpoint_path: Path to save/load library state
            max_sketches: Maximum sketches to keep (oldest removed when exceeded)
        """
        self.checkpoint_path = checkpoint_path
        self.max_sketches = max_sketches
        self.sketches: List[Dict] = []
        self.usage_stats: Dict[str, int] = {}  # Track how often each sketch helps
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load()
    
    def add_sketch(self, sketch: Dict) -> bool:
        """
        Add a new sketch to the library.
        
        Args:
            sketch: Sketch dictionary with id, trigger, template, slots
            
        Returns:
            True if added, False if duplicate
        """
        sketch_id = sketch.get('id', '')
        
        # Check for duplicates by pattern
        pattern = sketch.get('operation_pattern', sketch_id)
        existing_patterns = [s.get('operation_pattern', s.get('id', '')) for s in self.sketches]
        
        if pattern in existing_patterns:
            # Update existing instead of adding duplicate
            for i, s in enumerate(self.sketches):
                if s.get('operation_pattern', s.get('id', '')) == pattern:
                    s['confidence'] = str(max(
                        float(s.get('confidence', '0.5')),
                        float(sketch.get('confidence', '0.5'))
                    ))
                    s['updated_at'] = datetime.now().isoformat()
                    return False
        
        # Add new sketch
        sketch['added_at'] = datetime.now().isoformat()
        sketch['usage_count'] = 0
        self.sketches.append(sketch)
        
        # Trim if over max
        if len(self.sketches) > self.max_sketches:
            # Remove lowest confidence + lowest usage
            self.sketches.sort(key=lambda s: (
                float(s.get('confidence', '0')),
                s.get('usage_count', 0)
            ))
            self.sketches = self.sketches[1:]  # Remove lowest
        
        return True
    
    def add_sketches_from_rules(self, rules: List[Dict]):
        """
        Add sketch rules from library learner output.
        
        Args:
            rules: List of rule dictionaries from LibraryLearner
        """
        for rule in rules:
            if rule.get('has_sketch') and rule.get('sketch'):
                self.add_sketch({
                    'id': rule.get('id', 'UNKNOWN'),
                    'trigger': rule.get('trigger', ''),
                    'template': rule['sketch'].get('template', ''),
                    'slots': rule['sketch'].get('slots', []),
                    'operations': rule['sketch'].get('operations', []),
                    'operation_pattern': '_'.join(rule['sketch'].get('operations', [])),
                    'confidence': rule.get('confidence', '0.5'),
                    'source': rule.get('source', 'library_learner')
                })
    
    def get_sketches_for_question(self, question: str) -> List[Dict]:
        """
        Get relevant sketches for a given question.
        
        Args:
            question: The question text
            
        Returns:
            List of relevant sketches, sorted by confidence
        """
        question_lower = question.lower()
        relevant = []
        
        for sketch in self.sketches:
            trigger = sketch.get('trigger', '')
            # Check if any trigger words match
            trigger_words = [w.strip() for w in trigger.lower().replace(' or ', '|').split('|')]
            
            if any(word in question_lower for word in trigger_words if len(word) > 2):
                relevant.append(sketch)
        
        # Sort by confidence
        relevant.sort(key=lambda s: float(s.get('confidence', '0')), reverse=True)
        
        return relevant
    
    def format_for_solver(self, question: str = None) -> str:
        """
        Format sketches as a prompt section for the solver.
        
        This is the key integration point - it creates a prompt section
        that shows the LLM how to use sketch templates.
        
        Args:
            question: Optional question to filter relevant sketches
            
        Returns:
            Formatted string to include in solver prompt
        """
        if not self.sketches:
            return ""
        
        # Get relevant sketches (or all if no question)
        if question:
            sketches_to_show = self.get_sketches_for_question(question)
            if not sketches_to_show:
                sketches_to_show = self.sketches[:3]  # Show top 3 if none match
        else:
            sketches_to_show = self.sketches
        
        if not sketches_to_show:
            return ""
        
        lines = [
            "",
            "## Learned Program Templates (Use These!)",
            "",
            "These templates have been learned from successful programs. When the question matches a trigger, use the template and fill in the $N slots with values from the context.",
            ""
        ]
        
        for i, sketch in enumerate(sketches_to_show[:5]):  # Max 5 sketches
            lines.append(f"### Template {i+1}: {sketch.get('id', 'Unknown')}")
            lines.append(f"**When to use**: {sketch.get('trigger', 'General calculation')}")
            lines.append(f"**Pattern**: `{sketch.get('template', '')}`")
            
            # Explain slots
            slots = sketch.get('slots', [])
            if slots:
                lines.append("**Slots to fill**:")
                for slot in slots:
                    lines.append(f"  - `{slot.get('id', '')}`: {slot.get('semantic', 'value')} (extract from context)")
            
            # Show example instantiation
            ops = sketch.get('operations', [])
            if ops:
                if len(ops) == 1:
                    lines.append(f"**Example**: `[\"{ops[0]}(\", \"$0_value\", \"$1_value\", \")\", \"EOF\"]`")
                elif len(ops) == 2:
                    lines.append(f"**Example**: `[\"{ops[0]}(\", \"$0_value\", \"$1_value\", \")\", \"{ops[1]}(\", \"#0\", \"$1_value\", \")\", \"EOF\"]`")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def record_usage(self, sketch_id: str, was_helpful: bool):
        """
        Record whether a sketch was helpful for a prediction.
        
        Args:
            sketch_id: ID of the sketch
            was_helpful: Whether the prediction was correct
        """
        for sketch in self.sketches:
            if sketch.get('id') == sketch_id:
                if was_helpful:
                    sketch['usage_count'] = sketch.get('usage_count', 0) + 1
                break
    
    def save(self):
        """Save library to checkpoint file."""
        if not self.checkpoint_path:
            return
        
        data = {
            'sketches': self.sketches,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load library from checkpoint file."""
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            return
        
        with open(self.checkpoint_path, 'r') as f:
            data = json.load(f)
        
        self.sketches = data.get('sketches', [])
    
    def __len__(self) -> int:
        return len(self.sketches)
    
    def get_stats(self) -> Dict:
        """Get library statistics."""
        return {
            'total_sketches': len(self.sketches),
            'patterns': [s.get('operation_pattern', s.get('id', '')) for s in self.sketches],
            'avg_confidence': sum(float(s.get('confidence', '0')) for s in self.sketches) / len(self.sketches) if self.sketches else 0
        }


# Test the sketch library
if __name__ == "__main__":
    print("Testing Sketch Library...")
    print("=" * 60)
    
    library = SketchLibrary(max_sketches=5)
    
    # Add some test sketches
    library.add_sketch({
        'id': 'LEARNED_SUBTRACT_DIVIDE',
        'trigger': 'growth rate OR percentage change OR increase',
        'template': 'subtract($0:new_value, $1:old_value), divide(#0, $1:old_value)',
        'slots': [
            {'id': '$0', 'semantic': 'new_value'},
            {'id': '$1', 'semantic': 'old_value'}
        ],
        'operations': ['subtract', 'divide'],
        'confidence': '0.8'
    })
    
    library.add_sketch({
        'id': 'LEARNED_DIVIDE',
        'trigger': 'percentage of OR portion OR fraction',
        'template': 'divide($0:part, $1:total)',
        'slots': [
            {'id': '$0', 'semantic': 'part'},
            {'id': '$1', 'semantic': 'total'}
        ],
        'operations': ['divide'],
        'confidence': '0.7'
    })
    
    print(f"Library size: {len(library)}")
    print(f"Stats: {library.get_stats()}")
    print()
    
    # Test formatting for solver
    question = "What is the growth rate from 100 to 150?"
    print(f"Question: {question}")
    print()
    print("Solver prompt section:")
    print(library.format_for_solver(question))
    print()
    
    # Test filtering
    question2 = "What percentage of total revenue is domestic?"
    print(f"Question: {question2}")
    relevant = library.get_sketches_for_question(question2)
    print(f"Relevant sketches: {[s['id'] for s in relevant]}")
    
    print("=" * 60)
    print("✅ Sketch Library tests complete!")
