import datasetCreator
import torch
from torch import nn

BAD_PC = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

learning_rate = 1e-3
epochs = 5

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = None
        if (BAD_PC):
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(6291456, 100),
            nn.ReLU(),         
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,10)       
            )
        else:
            self.linear_relu_stack = nn.Sequential(
            nn.Linear(2097152, 1000),
            nn.ReLU(),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,10) 
            )
            

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

nextData = next(iter(datasetCreator.bremenTrainingLoader))

logits = model(nextData.float())
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")