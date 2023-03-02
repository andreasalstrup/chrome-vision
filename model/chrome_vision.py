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
      
class ChromeCut(nn.Module):
      def __init__(self, base_encoder, feature_dim=128, queue_size=65536, momentum=0.999, softmax_temp=0.07, mlp=False):
            """
            # feature_dim: uique classes in the target dataset
            # queue_size: number of keys in queue
            # momentum: controls the rate when applying 'momentum update'. Large value, measn smaller update.
            # softmat_temp: normalize the contrastive loss
            # mlp: multilayer perceptron
            """

            super(ChromeCut, self).__init__()

            self.queue_size = queue_size
            self.momentum = momentum
            self.softmax_temp = softmax_temp

            # Create query and key encoder
            self.encoder_query = base_encoder(num_classes=feature_dim)
            self.encoder_key = base_encoder(num_classes=feature_dim)

            # Adding MLP Projection Head for representation
            if mlp:
                  # Get the dimension of the first fully connected layer 
                  dim_mlp = self.encoder_query.fc.weight.shape[1]

                  # Setup layers in the query encoder
                  self.encoder_query.fc = nn.Sequential(
                        # Create a linear layer
                        nn.Linear(in_features=dim_mlp, 
                                  out_features=dim_mlp),
                        # max(0, out_features)
                        nn.ReLU(),
                        # A stack of fully connected layers
                        self.encoder_query.fc
                  )

                  # Setup layers in the key encoder
                  self.encoder_key.fc = nn.Sequential(
                        # Create a linear layer
                        nn.Linear(in_features=dim_mlp,
                                  out_features=dim_mlp),
                        # max(0, out_features)
                        nn.ReLU(),
                        # A stack of fully connected layers
                        self.encoder_key.fc
                  )
            
            # Sequence query encoder and key encoder as tuples
            for param_query, param_key in zip(self.encoder_query.parameters(), self.encoder_key.parameters()):
                  # Initialize key encoder with data from corresponding query encoder 
                  param_key.data.copy_(param_query.data)
                  # Initilaze key encoder to not have its gradients computed during backpropagation
                  param_key.requires_grad = False

            # Create queue
            # Initialize with random numbers
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            # Normalize all tensors in the queue
            self.queue = nn.functional.normalize(input=feature_dim, feature_dim=0)

            # Create queue pointer
            self.register_buffer("queue_ptr", torch.zeros(size=1, dtype=torch.long))
      
      # Take gradients from encoder_query and update parameters in the encoder_key
      # Make the key encoder processively evolving
      # Makes momentum contrast (MoCo) more memory efficient
      def momentum_update(self):
            # Sequence query encoder and key encoder as tuples
            for param_query, param_key in zip(self.encoder_query, self.encoder_key):
                  # Apply momentum update for all tuples
                  param_key.data = self.momentum * param_key.data + (1 - self.momentum) * param_query.data

      # Maintain the dictionary as a queue of data samples
      # The current mini-batch is enqueued to the dictionary and the oldest mini-batch in the queue is removed
      def enqueue_dequeue(self, key):
            # Get all keys in queue
            keys = get_keys(key)

            batch_size = keys.shape[0]

            current_ptr = int(self.queue_ptr)

            start_index = current_ptr
            end_index = current_ptr + batch_size
            # Preform enqueue and dequeue
            # Set the new queue. Get all rows
            self.queue[:, start_index : end_index] = keys.softmax_temp

            # Move the pointer
            new_ptr = (current_ptr + batch_size) % self.queue_size
            
            # Set the new pointer
            self.queue_ptr[0] = new_ptr


      def forward(self, query_batch_images, key_batch_images):
            logits = 0
            labels = 0
            return logits, labels
      
def get_keys(tensor):

      # Initialize all tensor element to one
      tensor_init_one = torch.ones_like(tensor)

      all_tensors = [tensor_init_one for _ in range(torch.distributed.get_world_size())]

      for _ in range(torch.distributed.get_world_size()):
            torch.ones_like(tensor)

      torch.distributed.all_gather(all_tensors, tensor, async_op=False)

      return torch.cat(all_tensors, dim=0)