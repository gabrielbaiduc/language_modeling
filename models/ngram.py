import torch
from .base import BaseLanguageModel

class NGramModel(BaseLanguageModel):
    def __init__(self, N: int):
        super().__init__()
        assert N >= 1, "N must be at least 1"
        self.N = N
        self.N_counts = None
        self.P = None

    def train(self, words: list[str]) -> None:
        """Train the NGram model on the given words."""
        # Build vocabulary first
        self.build_vocabulary(words)
        
        # Count N-grams
        shape = tuple([self.vocab_size] * self.N)
        self.N_counts = torch.zeros(shape, dtype=torch.int16)
        
        for w in words:
            chs = ['.'] * (self.N-1) + list(w) + ['.']
            for gram in zip(*[chs[i:] for i in range(self.N)]):
                idxs = [self.stoi[ch] for ch in gram]
                self.N_counts[tuple(idxs)] += 1

        # Compute probabilities
        self.P = self.N_counts.float()
        # Normalize along the last axis
        self.P /= self.P.sum(dim=-1, keepdim=True)
        # Replace NaNs (from division by zero) with zeros
        self.P = torch.nan_to_num(self.P, nan=0.0)
        self.P = self.P.to(torch.float16)
        
        self.is_trained = True

    def sample(self, num_samples: int = 1, seed: int | None = None) -> list[str]:
        """Generate samples from the trained NGram model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before sampling")
            
        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
        
        results = []
        for _ in range(num_samples):
            out = []
            context = [0] * (self.N - 1)
            while True:
                p = self.P[tuple(context)]
                if g is not None:
                    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                else:
                    ix = torch.multinomial(p, num_samples=1, replacement=True).item()
                out.append(self.itos[ix])
                if ix == 0:
                    break
                context = context[1:] + [ix] if self.N > 1 else []
            results.append(''.join(out))
        return results

    def evaluate(self, words: list[str]) -> dict[str, float]:
        """Evaluate the negative log-likelihood of the model on the given words."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
            
        log_likelihood = 0.0
        n = 0
        
        for w in words:
            chs = ['.'] * (self.N - 1) + list(w) + ['.']
            for gram in zip(*[chs[i:] for i in range(self.N)]):
                idxs = [self.stoi[ch] for ch in gram]
                prob = self.P[tuple(idxs)]
                # Avoid log(0) by clamping probability to a small positive value
                prob = max(float(prob), 1e-10)
                logprob = torch.log(torch.tensor(prob))
                log_likelihood += logprob
                n += 1
                
        nll = -log_likelihood
        avg_nll = nll / n if n > 0 else float('inf')
        
        return {
            "log_likelihood": float(log_likelihood),
            "nll": float(nll),
            "avg_nll": float(avg_nll),
            "num_ngrams": n
        }