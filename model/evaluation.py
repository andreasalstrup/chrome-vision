import torch
from torch import device, nn
from tqdm.auto import tqdm
from model.utilis import accuracy_fn, accuracy_top_k

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, 
               accuracy_fn=accuracy_fn,
               device: torch.device = device):
   
   train_loss, train_acc = 0, 0
    
   model.train()

   # X: image (features)
   for batch, (X, _) in enumerate(data_loader):
      # put data on target device       
      X = X.to(device)

      # 1. Forward pass
      output, target = model(query_batch_images=X[0],key_batch_images=X[1])

      # 2. Calculate loss (per batch)
      loss = nn.CrossEntropyLoss(output, target)
      #acc1, acc5 = accuracy_top_k(output, target, topk=(1, 5))
      
      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

   # Divide total train loss by length of train dataloader (average loss per batch per epoch)
   train_loss /= len(data_loader)
   train_acc /= len(data_loader)

   if batch % 10 == 0:
      print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%')


def train_step_label(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, 
               accuracy_fn=accuracy_fn,
               device: torch.device = device):
   
   train_loss, train_acc = 0, 0
    
   model.train()

   # X: image (features), y: label
   for batch, (X, y) in enumerate(data_loader):
        # put data on target device        
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # acc train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Logits -> prediction labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

   # Divide total train loss by length of train dataloader (average loss per batch per epoch)
   train_loss /= len(data_loader)
   train_acc /= len(data_loader)

   print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%')

def test_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device = device):
   
   loss, acc = 0, 0
   
   model.eval()

   with torch.inference_mode():
      for X_test, y_test in tqdm(data_loader):
         X_test, y_test = X_test.to(device), y_test.to(device)
         # Make predictions
         y_pred = model(X_test)

         # Calculate loss 
         loss += loss_fn(y_pred, y_test)

         # Calculate accuracy
         acc += accuracy_fn(y_true=y_test, y_pred=y_pred.argmax(dim=1))

      # Divide total train loss by length of train dataloader (average loss per batch per epoch)
      loss /= len(data_loader)
      acc /= len(data_loader)

   return {"model_name": model.__class__.__name__, 
           "model_loss": loss.item(),
           "model_acc": acc}