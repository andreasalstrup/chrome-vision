import torch
from torch import nn

### Material
# https://www.learnpytorch.io/01_pytorch_workflow/#2-build-model
# https://pytorch.org/tutorials/beginner/ptcheat.html

### The main algorithms (tbd):
# Momentum update

# Create a neural network module subclass
class ChromeVisionModel(nn.Module):
        
        # Initialize model parameters
        def __init__(self,
                     input_shape: int,
                     hidden_units: int,
                     output_shape: int):

            super().__init__()

            self.layer_stack = nn.Sequential(
                # 1. Pass sample through the flatten layer
                nn.Flatten(),
                # 2. Pass output of flatten layer to a linear layer
                nn.Linear(in_features=input_shape,
                          out_features=hidden_units),
                # 3. Wont change shape of data
                nn.ReLU(),
                # 4. Pass output of ReLU layer to a linear layer 
                nn.Linear(in_features=hidden_units,
                          out_features=output_shape),
                # 5.
                nn.ReLU()
            )
        
        # Forwad propagation
        # Executed at every call
        # First call is the model call -> model(input)
        # Models take input x (one batch at a time)
        def forward(self, x: torch.Tensor): 
              return self.layer_stack(x)