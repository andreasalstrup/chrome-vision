import torch
from torch import nn

### Material
# https://www.learnpytorch.io/01_pytorch_workflow/#2-build-model
# https://pytorch.org/tutorials/beginner/ptcheat.html

### The main algorithms (tbd):
# Momentum update

# Create a neural network module subclass
class ChromeMoCo(nn.Module):
      def __init__(self, base_encoder, feature_dim=128, queue_size=65536, momentum=0.999, softmax_temp=0.07):
            """
            # feature_dim: uique classes in the target dataset
            # queue_size: number of keys in queue
            # momentum: controls the rate when applying 'momentum update'. Large value, measn smaller update.
            # softmat_temp: normalize the contrastive loss
            # mlp: multilayer perceptron
            """

            super(ChromeMoCo, self).__init__()

            self.queue_size = queue_size
            self.momentum = momentum
            self.softmax_temp = softmax_temp

            # Create query and key encoder
            self.encoder_query = base_encoder(weights=None, num_classes=feature_dim)
            self.encoder_key = base_encoder(weights=None, num_classes=feature_dim)

            # Adding MLP Projection Head for representation
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
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
      
      # Take gradients from encoder_query and update parameters in the encoder_key
      # Make the key encoder processively evolving
      # Makes momentum contrast (MoCo) more memory efficient
      @torch.no_grad()
      def momentum_update(self):
            # Sequence query encoder and key encoder as tuples
            for param_query, param_key in zip(self.encoder_query.parameters(), self.encoder_key.parameters()):
                  # Apply momentum update for all tuples
                  param_key.data = self.momentum * param_key.data + (1 - self.momentum) * param_query.data

      # Maintain the dictionary as a queue of data samples
      # The current mini-batch is enqueued to the dictionary and the oldest mini-batch in the queue is removed
      @torch.no_grad()
      def enqueue_dequeue(self, keys):
            batch_size = keys.shape[0]

            current_ptr = int(self.queue_ptr)
            assert self.queue_size % batch_size == 0, f"Queue size ({self.queue_size}) is not a factor of Batch size ({batch_size})"

            start_index = current_ptr
            end_index = current_ptr + batch_size

            # Preform enqueue and dequeue
            # Set the new queue. Get all rows
            self.queue[:, start_index : end_index] = keys.T

            # Move the pointer
            new_ptr = (current_ptr + batch_size) % self.queue_size
            
            # Set the new pointer
            self.queue_ptr[0] = new_ptr

      def forward(self, query_batch_images, key_batch_images):

            query = self.encoder_query(query_batch_images)
            query = nn.functional.normalize(query, dim=1)

            with torch.no_grad():
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
            
            # Apply softmax temperature scaling
            # Deviding softmax_temp with each value in tensor  
            logits /= self.softmax_temp

            # Instantiate a tensor of zeros of the size logits.shape[0] (rows in logits = number of examples in batch)
            if torch.cuda.is_available():
                  labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            else:
                  labels = torch.zeros(logits.shape[0], dtype=torch.long)

            self.enqueue_dequeue(keys)

            return logits, labels