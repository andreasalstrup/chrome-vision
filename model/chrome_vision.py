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
            self.encoder_query = base_encoder(in_features=784, num_classes=feature_dim)
            print(self.encoder_query)
            self.encoder_key = base_encoder(in_features=784, num_classes=feature_dim)
            print(self.encoder_key)

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
            self.queue = nn.functional.normalize(input=self.queue, dim=0)

            # Create queue pointer
            self.register_buffer("queue_ptr", torch.zeros(1))
      
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
      def enqueue_dequeue(self, keys):
            # Get all keys in queue
            #keys = get_distributed_keys(keys)

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

            query = self.encoder_query(query_batch_images)
            query = nn.functional.normalize(query, dim=1)

            with torch.inference_mode():
                  self.momentum_update()

                  keys = self.encoder_key(key_batch_images)
                  keys = nn.functional.normalize(keys, dim=1)
            
            # Compute logits using the dot product
            # We measure the similarity (distance) between too vectors
            # Results in a tenstor, containing the score for each position
            # Free Indices: Specified in the output
            # Summation Indices: Indices in the input argument but not in output specification
            
            # for n in range(query size)
            #     total = 0
            #     for c in range(keys size)
            #           total += query[n,c]*keys[n,c]
            #
            #     positive_logits[n] = total
            #
            # Free Indices: n
            # Summation Indices: c
            #
            # Output dimension: Nx1
            positive_logits = torch.einsum("nc,nc->n", [query, keys])

            # Add a new dimentions at then end of the tensor and make it a 2D tensor
            positive_logits = positive_logits.unsqueeze(dim=-1)

            # Create a clone of the query thereby not affecting the original tensor or its gradients
            query_clone = self.queue.clone().detach()

            # for n in range(dim)
            #     for k in range(dim)
            #           total = 0
            #           for c in range(dim)
            #                 total += query[n,c]*query_clone[c,k]
            #
            #     negative_logits[n,k] = total
            #
            # Free Indices: nk
            # Summation Indices: c
            #
            # Output dimension: NxK
            negative_logits = torch.einsum("nc,ck->nk", [query, query_clone])

            # Concatenate positive_logits and negative_logits along the secound dimension (columns)
            # Output dimension: Nx(1+K), (rows)x(columns)
            logits = torch.cat([positive_logits, negative_logits], dim=1)

            print(f'positive_logits shape: {positive_logits.shape}\n')
            print(f'negative_logits shape: {negative_logits.shape}\n')
            print(f'Logits shape: {logits.shape}\n')
            
            # Apply softmax temperature scaling
            # Deviding softmax_temp with each value in tensor  
            logits /= self.softmax_temp

            # Instantiate a tensor of zeros of the size logits.shape[0] (rows in logits = number of examples in batch)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            self.enqueue_dequeue(keys)

            return logits, labels

@torch.inference_mode()      
def get_distributed_keys(tensor):
      # Create a new tensor of the same shape and data type as the input tensor but with all elements initialized to 1
      tensor_init_one = [torch.ones_like(tensor)]

      # Create a list with a length determined by the number of processes in the distributed environment
      tensors = []
      for _ in range(torch.distributed.get_world_size()):
            tensors.append(tensor_init_one)

      # Stores the tensors that are gathered from all processes
      torch.distributed.all_gather(tensors, tensor, async_op=False)

      # Concatenated tensors to a single tensor
      return torch.cat(tensors, dim=0)