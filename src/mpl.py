
import torch

from .base import BaseLanguageModel

class MLPModel(BaseLanguageModel):
    def __init__(self, block_size: int):
        self.block_size = block_size

    def build_dataset(self, words: list[str]):  
        X, Y = [], []  # Input contexts and target characters

        for w in words:
            # Start with context of zeros (padding): [0, 0, 0] for block_size=3
            context = [0] * self.block_size
            
            for ch in w + '.':  # Process each char + end token '.'
                ix = self.stoi[ch]  # Convert char to index
                # Example with "emma" and block_size=3:
                # context=[0,0,0] → predict 'e' (ix=5)
                # context=[0,0,5] → predict 'm' (ix=13) 
                # context=[0,5,13] → predict 'm' (ix=13)
                # context=[5,13,13] → predict 'a' (ix=1)
                # context=[13,13,1] → predict '.' (ix=0)
                X.append(context)       # Current context as input
                Y.append(ix)           # Next character as target
                context = context[1:] + [ix]  # Slide window: drop first, add current

        X = torch.tensor(X)  # Shape: [num_examples, block_size]
        Y = torch.tensor(Y)  # Shape: [num_examples]
        print(X.shape, Y.shape)
        return X, Y

    def train(self):
        pass

    def sample(self):
        pass

    def evaluate(self):
        pass