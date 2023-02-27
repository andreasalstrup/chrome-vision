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
        
# A Convolutional Neural network
# https://poloclub.github.io/cnn-explainer/
class ChromeVisionModelV2(nn.Module):
      def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
            
            super().__init__()

            # Feature extractor layers
            self.conv_block_1 = nn.Sequential(
                  nn.Conv2d(in_channels=input_shape,
                            out_channels=hidden_units,
                            kernel_size=3,
                            stride=1,
                            padding=1),
                   nn.ReLU(),
                   nn.Conv2d(in_channels=hidden_units,
                             out_channels=hidden_units,
                             kernel_size=3,
                             stride=1,
                             padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2) # Take max value in window
            )

            # Feature extractor layers
            self.conv_block_2 = nn.Sequential(
                  nn.Conv2d(in_channels=hidden_units,
                            out_channels=hidden_units,
                            kernel_size=3,
                            stride=1,
                            padding=1),
                   nn.ReLU(),
                   nn.Conv2d(in_channels=hidden_units,
                             out_channels=hidden_units,
                             kernel_size=3,
                             stride=1,
                             padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
            )

            # Classifier layers
            self.classifier = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(in_features=hidden_units*7*7, # Output shape of conv_block_2
                            out_features=output_shape)  # outpu_chape: each class
            )

      def forward(self, x):
            x = self.conv_block_1(x)
            #print(x.shape)
            x = self.conv_block_2(x)
            #print(x.shape)
            x = self.classifier(x)
            return x