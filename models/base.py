import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import random

class BaseLanguageModel(ABC):
    def __init__(self):
        self.vocab_size = None
        self.stoi = None
        self.itos = None
        self.is_trained = False
    
    def load_data(self, filepath: str) -> List[str]:
        """Load words from a text file (names.txt or names_complex.txt format)."""
        with open(filepath, 'r') as f:
            words = f.read().strip().split('\n')
        return [w.strip().lower() for w in words if w.strip()]
    
    def build_vocabulary(self, words: List[str]):
        """Build character vocabulary from words."""
        chars = sorted(list(set(''.join(words))))
        self.stoi = {s: i+1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0  # Special start/end token
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
    
    def split_data(self, words: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1, 
                   test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
        """Split words into train/validation/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        words_copy = words.copy()
        random.seed(seed)
        random.shuffle(words_copy)
        
        n = len(words_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_words = words_copy[:train_end]
        val_words = words_copy[train_end:val_end]
        test_words = words_copy[val_end:]
        
        return train_words, val_words, test_words
    
    @abstractmethod
    def train(self, words: List[str]) -> None:
        """Train the model on the given words."""
        pass
    
    @abstractmethod
    def sample(self, num_samples: int = 1, seed: Optional[int] = None) -> List[str]:
        """Generate samples from the trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, words: List[str]) -> Dict[str, float]:
        """Evaluate the model on the given words."""
        pass 