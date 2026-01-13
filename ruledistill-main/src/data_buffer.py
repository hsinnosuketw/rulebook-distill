"""
Data Buffer

Manages dataset iteration and batch processing for the self-regulating
rule augmentation pipeline.
"""

import json
import random
from typing import Iterator, Optional


class DataBuffer:
    """
    Feeds questions in batches, pairing each with ground truth.
    
    Handles:
    - Dataset loading from FinQA JSON format
    - Batch iteration with configurable size
    - Optional shuffling for training
    - Ground truth extraction from programs or direct answers
    """
    
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 10,
        shuffle: bool = False,
        limit: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the data buffer.
        
        Args:
            dataset_path: Path to FinQA JSON dataset
            batch_size: Number of questions per batch
            shuffle: Whether to shuffle the dataset
            limit: Optional limit on number of samples to use
            seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Load dataset
        self.dataset = self._load_dataset(dataset_path)
        
        # Apply limit if specified
        if limit and limit < len(self.dataset):
            self.dataset = self.dataset[:limit]
        
        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.dataset)
        
        self.current_idx = 0
        self.total_batches = (len(self.dataset) + batch_size - 1) // batch_size
    
    def _load_dataset(self, path: str) -> list[dict]:
        """Load FinQA dataset from JSON file."""
        print(f"Loading dataset from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples.")
        return data
    
    def _extract_item(self, item: dict, idx: int) -> dict:
        """
        Extract question, context, and ground truth from dataset item.
        
        Args:
            item: Raw dataset item
            idx: Index in the dataset (for tracking)
            
        Returns:
            Dictionary with question, context, ground_truth, program, and metadata
        """
        qa = item.get('qa', {})
        
        # Extract question
        question = qa.get('question', '')
        
        # Extract context from gold_inds
        gold_inds = qa.get('gold_inds', {})
        if isinstance(gold_inds, dict):
            context = " ".join(gold_inds.values())
        elif isinstance(gold_inds, list):
            context = " ".join([str(x) for x in gold_inds])
        else:
            context = str(gold_inds)
        
        # Extract ground truth - use exe_ans (program execution result) for accuracy
        program = qa.get('program', '')
        # Prefer exe_ans (computed from program) over raw answer for better accuracy
        answer = qa.get('exe_ans', qa.get('answer', ''))
        
        return {
            "idx": idx,
            "question": question,
            "context": context,
            "ground_truth": answer,
            "program": program,
            "raw_item": item  # Keep for reference if needed
        }
    
    def __iter__(self) -> Iterator[list[dict]]:
        """Iterate over batches of prepared items."""
        self.current_idx = 0
        
        while self.current_idx < len(self.dataset):
            batch = self.get_batch(self.current_idx)
            self.current_idx += self.batch_size
            yield batch
    
    def __len__(self) -> int:
        """Return total number of batches."""
        return self.total_batches
    
    def get_batch(self, start_idx: int) -> list[dict]:
        """
        Get a specific batch by starting index.
        
        Args:
            start_idx: Starting index in the dataset
            
        Returns:
            List of prepared item dictionaries
        """
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        
        batch = []
        for i in range(start_idx, end_idx):
            item = self._extract_item(self.dataset[i], i)
            batch.append(item)
        
        return batch
    
    def get_batch_by_number(self, batch_num: int) -> list[dict]:
        """
        Get a batch by its batch number (0-indexed).
        
        Args:
            batch_num: Batch number (0-indexed)
            
        Returns:
            List of prepared item dictionaries
        """
        start_idx = batch_num * self.batch_size
        return self.get_batch(start_idx)
    
    def reset(self, shuffle: bool = False):
        """
        Reset the buffer to the beginning.
        
        Args:
            shuffle: Whether to reshuffle the dataset
        """
        self.current_idx = 0
        if shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.dataset)
    
    @property
    def remaining_batches(self) -> int:
        """Return number of remaining batches."""
        remaining_items = len(self.dataset) - self.current_idx
        return (remaining_items + self.batch_size - 1) // self.batch_size
    
    @property
    def total_items(self) -> int:
        """Return total number of items in the dataset."""
        return len(self.dataset)
    
    def get_stats(self) -> dict:
        """Return statistics about the data buffer."""
        return {
            "total_items": len(self.dataset),
            "batch_size": self.batch_size,
            "total_batches": self.total_batches,
            "current_batch": self.current_idx // self.batch_size,
            "remaining_batches": self.remaining_batches,
        }
