
import torch
import torch.nn.functional as F

from .base import BaseLanguageModel
from .utils.decay import step_decay_lr

class MLPModel(BaseLanguageModel):
    def __init__(self, block_size: int, n_embd: int, n_hidden: int):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.lossi = []
        self._init_params()
        self._init_grads()

    def _init_grads(self):
        self.parameters = [self.W1, self.W2, self.b1, self.b2]
        print(f'Total number of parameters: {sum(p.nelement() for p in self.parameters)}') # number of parameters in total
        for p in self.parameters:
            p.requires_grad = True

    def _init_params(self):
        self.C = torch.randn((self.vocab_size, self.n_embd))
        self.W1 = torch.randn((self.n_embd * self.block_size, self.n_hidden))
        self.b1 = torch.randn(self.n_hidden)
        self.W2 = torch.randn((self.n_hidden, self.vocab_size))
        self.b2 = torch.randn(self.vocab_size)

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

    def _forward(self, X):
        """Private method to perform forward pass"""
        emb = self.C[X]  # embed the characters into vectors
        embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors
        # Linear layer
        hpreact = embcat @ self.W1 + self.b1  # hidden layer pre-activation
        # Non-linearity
        h = torch.tanh(hpreact)  # hidden layer
        logits = h @ self.W2 + self.b2  # output layer
        return logits

    def train(self, words: list[str], max_steps: int, batch_size: int, 
              initial_lr: float = 0.1, decay_steps: int = 10000, decay_factor: float = 0.1):
        Xtr, Ytr = self.build_dataset(words)
        for i in range(max_steps):
            # minibatch construct
            ix = torch.randint(0, Xtr.shape[0], (batch_size,))
            Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
            
            # forward pass
            logits = self._forward(Xb)
            loss = F.cross_entropy(logits, Yb) # loss function
            
            # backward pass
            for p in self.parameters:
                p.grad = None
            loss.backward()
            
            # update with step decay
            lr = step_decay_lr(initial_lr, i, decay_steps, decay_factor)
            for p in self.parameters:
                p.data += -lr * p.grad

            # track stats
            if i % 10000 == 0: # print every once in a while
                print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}, lr: {lr:.6f}')
            self.lossi.append(loss.log10().item())

    def sample(self, num_samples: int = 1, seed: int | None = None) -> list[str]:
        """Generate samples from the trained MLP model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before sampling")
            
        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
        
        results = []
        for _ in range(num_samples):
            out = []
            context = [0] * self.block_size  # Start with padding tokens
            
            while True:
                # Convert context to tensor and get logits
                X = torch.tensor([context])  # Shape: [1, block_size]
                logits = self._forward(X)    # Shape: [1, vocab_size]
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=1)[0]  # Shape: [vocab_size]
                
                # Sample from the probability distribution
                if g is not None:
                    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                else:
                    ix = torch.multinomial(probs, num_samples=1, replacement=True).item()
                
                # Add character to output
                out.append(self.itos[ix])
                
                # Stop if we hit the end token
                if ix == 0:
                    break
                
                # Update context: slide window (drop first, add current)
                context = context[1:] + [ix]
            
            results.append(''.join(out))
        return results

    def evaluate(self, words: list[str]):
        # Build dataset from the evaluation words
        X_eval, Y_eval = self.build_dataset(words)
        
        # Forward pass using the private method
        logits = self._forward(X_eval)
        loss = F.cross_entropy(logits, Y_eval)
        
        return loss.item()