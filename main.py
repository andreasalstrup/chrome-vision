import torch
from torch import device, nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from model.evaluation import train_step, test_step, accuracy_fn # use torchmetrics.Accuracy()
from model.utilis import print_train_time
from model.chrome_vision import ChromeVisionModel
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

### The main algorithms (tbd):
# Softmax loss function

print(f'PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

# Data loading
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

class_names = train_data.classes

BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

print(f'Len of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}')
print(f'Len of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}')

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

## Turn data into batches (mini-batches)
# More computationally efficent, so we dont store all data in memory
# Gives neural network more changes to update its gradients per epoch
# andrew ng minibatches

### Baseline model

# Create a flatten layer
flatten_model = nn.Flatten()

x = train_features_batch[0]
#print(x)

# Flatten the sample - Turn into a vector, one value per pixel
output = flatten_model(x)
print(f'Shape before flattening: {x.shape}')     # torch.Size([1, 28, 28])
print(f'Shape after flattening: {output.shape}') # torch.Size([1, 784])

model = ChromeVisionModel(
    input_shape=784,    # 28*28
    hidden_units=10,    # Units in the hidden layer
    output_shape=len(class_names) # one for every class
).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# Training loop
torch.manual_seed(42)
train_time_start_on_cpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n')

    train_step(model=model,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    
    test_step(model=model,
               data_loader=test_dataloader,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn,
               device=device)

# Print time taken
train_time_end_on_cpu = timer()
total_train_time_model = print_train_time(train_time_start_on_cpu, train_time_end_on_cpu, str(next(model.parameters()).device))

# Calculate model results on test dataset
model_results = test_step(model=model,
                           data_loader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn,
                           device=device)

print(model_results)