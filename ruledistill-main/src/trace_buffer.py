"""
Trace Buffer Module

Stores successful program traces during the "Wake" phase for later
abstraction during the "Sleep" phase. Part of Wake-Sleep Library Learning.

Based on DreamCoder research for neuro-symbolic program synthesis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime


@dataclass
class BufferedTrace:
    """A successful trace stored in the buffer."""
    question: str
    program: List[str]
    result: Any
    operation_pattern: str  # e.g., "subtract_divide" for clustering
    arg_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


class TraceBuffer:
    """
    Buffer for storing successful program traces during Wake phase.
    
    Key features:
    - Stores only successful traces (correct answers)
    - Tracks operation patterns for clustering
    - Provides traces for Sleep phase abstraction
    - Maintains sliding window to prevent unbounded growth
    """
    
    def __init__(self, max_traces: int = 1000, checkpoint_path: str = None):
        """
        Initialize the trace buffer.
        
        Args:
            max_traces: Maximum number of traces to store (FIFO eviction)
            checkpoint_path: Optional path to save/load buffer state
        """
        self.max_traces = max_traces
        self.checkpoint_path = checkpoint_path
        self.traces: List[BufferedTrace] = []
        self.pattern_index: Dict[str, List[int]] = defaultdict(list)
        
        # Load existing checkpoint if available
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint()
    
    def _extract_pattern(self, program: List[str]) -> str:
        """Extract operation pattern from program for clustering."""
        ops = []
        for i, token in enumerate(program):
            if i % 4 == 0 and token != "EOF":
                # Extract operation name without the "("
                op = token.rstrip("(")
                ops.append(op)
        return "_".join(ops)
    
    def _count_args(self, program: List[str]) -> int:
        """Count number of arguments (non-reference operands) in program."""
        count = 0
        for i, token in enumerate(program):
            if i % 4 in [1, 2]:  # arg1 or arg2 positions
                if token != "EOF" and not token.startswith("#"):
                    count += 1
        return count
    
    def add_success(
        self, 
        question: str,
        program: List[str],
        result: Any,
        metadata: Dict = None
    ) -> bool:
        """
        Store a successful program trace.
        
        Args:
            question: The question that was answered correctly
            program: The DSL program that produced correct answer
            result: The correct result
            metadata: Optional additional data (e.g., confidence, model)
            
        Returns:
            True if trace was added, False if duplicate
        """
        pattern = self._extract_pattern(program)
        arg_count = self._count_args(program)
        
        trace = BufferedTrace(
            question=question[:500],  # Truncate for storage
            program=program,
            result=result,
            operation_pattern=pattern,
            arg_count=arg_count,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self.traces.append(trace)
        self.pattern_index[pattern].append(len(self.traces) - 1)
        
        # Evict oldest if over capacity
        if len(self.traces) > self.max_traces:
            self._evict_oldest()
        
        return True
    
    def _evict_oldest(self):
        """Remove oldest traces to maintain max_traces limit."""
        while len(self.traces) > self.max_traces:
            removed = self.traces.pop(0)
            # Rebuild pattern index (expensive but infrequent)
            self._rebuild_pattern_index()
    
    def _rebuild_pattern_index(self):
        """Rebuild the pattern index after eviction."""
        self.pattern_index = defaultdict(list)
        for i, trace in enumerate(self.traces):
            self.pattern_index[trace.operation_pattern].append(i)
    
    def get_traces_by_pattern(self, pattern: str) -> List[BufferedTrace]:
        """Get all traces matching a specific operation pattern."""
        indices = self.pattern_index.get(pattern, [])
        return [self.traces[i] for i in indices if i < len(self.traces)]
    
    def get_pattern_counts(self) -> Dict[str, int]:
        """Get count of traces per operation pattern."""
        return {p: len(indices) for p, indices in self.pattern_index.items()}
    
    def get_clusterable_patterns(self, min_count: int = 5) -> List[str]:
        """
        Get operation patterns with enough traces for abstraction.
        
        Args:
            min_count: Minimum number of traces needed for clustering
            
        Returns:
            List of pattern strings with sufficient examples
        """
        counts = self.get_pattern_counts()
        return [p for p, c in counts.items() if c >= min_count]
    
    def get_all_traces(self) -> List[BufferedTrace]:
        """Get all buffered traces."""
        return self.traces.copy()
    
    def clear(self):
        """Clear all traces from buffer."""
        self.traces = []
        self.pattern_index = defaultdict(list)
    
    def save_checkpoint(self):
        """Save buffer state to checkpoint file."""
        if not self.checkpoint_path:
            return
        
        data = {
            "traces": [
                {
                    "question": t.question,
                    "program": t.program,
                    "result": t.result if not isinstance(t.result, float) or t.result == t.result else None,
                    "operation_pattern": t.operation_pattern,
                    "arg_count": t.arg_count,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata
                }
                for t in self.traces
            ],
            "saved_at": datetime.now().isoformat()
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_checkpoint(self):
        """Load buffer state from checkpoint file."""
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            return
        
        with open(self.checkpoint_path, 'r') as f:
            data = json.load(f)
        
        self.traces = []
        self.pattern_index = defaultdict(list)
        
        for t_data in data.get("traces", []):
            trace = BufferedTrace(
                question=t_data["question"],
                program=t_data["program"],
                result=t_data["result"],
                operation_pattern=t_data["operation_pattern"],
                arg_count=t_data["arg_count"],
                timestamp=t_data.get("timestamp", ""),
                metadata=t_data.get("metadata", {})
            )
            self.traces.append(trace)
            self.pattern_index[trace.operation_pattern].append(len(self.traces) - 1)
    
    def __len__(self) -> int:
        return len(self.traces)
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        pattern_counts = self.get_pattern_counts()
        return {
            "total_traces": len(self.traces),
            "unique_patterns": len(pattern_counts),
            "top_patterns": sorted(
                pattern_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "clusterable_patterns": len(self.get_clusterable_patterns())
        }


# Test the trace buffer
if __name__ == "__main__":
    print("Testing Trace Buffer...")
    print("=" * 60)
    
    buffer = TraceBuffer(max_traces=100)
    
    # Add some test traces
    test_programs = [
        (["subtract(", "100", "50", ")", "divide(", "#0", "50", ")", "EOF"], 1.0, "growth rate"),
        (["subtract(", "200", "150", ")", "divide(", "#0", "150", ")", "EOF"], 0.333, "percentage change"),
        (["subtract(", "500", "400", ")", "divide(", "#0", "400", ")", "EOF"], 0.25, "yoy change"),
        (["divide(", "100", "500", ")", "EOF"], 0.2, "percentage of total"),
        (["divide(", "50", "200", ")", "EOF"], 0.25, "portion"),
        (["add(", "100", "200", ")", "EOF"], 300, "sum values"),
        (["subtract(", "300", "100", ")", "divide(", "#0", "100", ")", "EOF"], 2.0, "increase rate"),
        (["subtract(", "400", "300", ")", "divide(", "#0", "300", ")", "EOF"], 0.333, "growth"),
    ]
    
    for prog, result, q in test_programs:
        buffer.add_success(f"What is the {q}?", prog, result)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Stats: {buffer.get_stats()}")
    print()
    
    print("Traces by pattern 'subtract_divide':")
    traces = buffer.get_traces_by_pattern("subtract_divide")
    for t in traces:
        print(f"  {t.program} -> {t.result}")
    print()
    
    print("Clusterable patterns (min 3):")
    print(buffer.get_clusterable_patterns(min_count=3))
    
    print("=" * 60)
    print("✅ Trace Buffer tests complete!")
