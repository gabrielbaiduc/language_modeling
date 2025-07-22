from abc import ABC, abstractmethod
import random
from pathlib import Path

import matplotlib.pyplot as plt

class BaseLanguageModel(ABC):
    def __init__(self, filename: str = 'names.txt'):
        self.words = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None
        self._load_data(filename)
        self._build_vocabulary()
        self.train_words = None
        self.val_words = None
        self.test_words = None
        self.is_trained = False
    
    def _load_data(self, filename: str, base_path: str = Path('data')) -> list[str]:
        """Load words from a text file (names.txt or names_complex.txt format)."""
        filepath = base_path / filename
        with open(filepath, 'r') as f:
            self.words = f.read().strip().split('\n')
        return [w.strip().lower() for w in self.words if w.strip()]
    
    def _build_vocabulary(self):
        """Build character vocabulary from words."""
        chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s: i+1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0  # Special start/end token
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
    
    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                   test_ratio: float = 0.1) -> tuple[list[str], list[str], list[str]]:
        """Split words into train/validation/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        words_copy = self.words.copy()
        random.shuffle(words_copy)
        
        n = len(words_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        self.train_words = words_copy[:train_end]
        self.val_words = words_copy[train_end:val_end]
        self.test_words = words_copy[val_end:]
        
        print(f'{len(self.train_words)=}, {len(self.val_words)=}, {len(self.test_words)=}')

    def plot(self):
        plt.plot(self.lossi)
        # Add running mean as a red curve
        running_mean = []
        cumsum = 0
        for i, loss in enumerate(self.lossi):
            cumsum += loss
            running_mean.append(cumsum / (i + 1))
        plt.plot(running_mean, color='red', label='Running Mean')
        plt.legend()
        plt.show()
    
    @abstractmethod
    def train(self, words: list[str]) -> None:
        """Train the model on the given words."""
        pass
    
    @abstractmethod
    def sample(self, num_samples: int = 1, seed: int | None = None) -> list[str]:
        """Generate samples from the trained model."""
        pass
    
    @abstractmethod
    def evaluate(self, words: list[str]) -> dict[str, float]:
        """Evaluate the model on the given words."""
        pass 