import torch

class RandomClassifier:
    def forward(self, x):
        return torch.randint(0, 2, (x.size,))

    def __call__(self, x):
        return self.forward(x)
