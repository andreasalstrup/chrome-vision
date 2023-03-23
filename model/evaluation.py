import torch
from torch import device, nn
from tqdm.auto import tqdm


def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, 
               accuracy_fn,
               device: torch.device = device):
   
   train_loss, train_acc1, train_acc5 = 0, 0, 0
    
   model.train()

   # X: image (features)
   for batch, images in enumerate(tqdm(data_loader)):

      # Add an extra dimension to image (tensor) in batch.
      # 3rd to 4th dimension
      # (1, C, H, W)  
      query_image = images[0].to(device)
      key_image = images[1].to(device)

      # 1. Forward pass
      output, target = model(query_batch_images=query_image.unsqueeze(0),key_batch_images=key_image.unsqueeze(0))
      output.requires_grad = True

      # 2. Calculate loss (per batch)
      loss = loss_fn(output, target)
      train_loss += loss

      acc1, acc5 = accuracy_fn(output, target, topk=(1, 5))      
      train_acc1 += acc1.item()
      train_acc5 += acc5.item()
      
      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # train_loss /= len(images)
      # train_acc1 /= len(images)
      # train_acc5 /= len(images)
      
   # Divide total train loss by length of train dataloader (average loss per batch per epoch)
   train_loss /= len(data_loader)
   train_acc1 /= len(data_loader)
   train_acc5 /= len(data_loader)

   print(f'Train loss: {train_loss:.5f} | Train acc1: {train_acc1:.2f}% | Train acc5: {train_acc5:.2f}%')
   return train_loss, train_acc5


def test_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device = device):
   
   test_loss, test_acc1, test_acc5 = 0, 0, 0
   
   model.eval()

   with torch.inference_mode():
      for batch, (images) in enumerate(tqdm(data_loader)):

         # Add an extra dimension to image (tensor) in batch.
         # 3rd to 4th dimension
         # (1, C, H, W)  
         query_image = images[0].unsqueeze(0).to(device)
         key_image = images[1].unsqueeze(0).to(device)

         # Make predictions
         output, target = model(query_batch_images=query_image,key_batch_images=key_image)

         # Calculate loss 
         test_loss += loss_fn(output, target)

         # Calculate accuracy
         acc1, acc5 = accuracy_fn(output, target, topk=(1, 5))
         test_acc1 += acc1.item()
         test_acc5 += acc5.item()

      # Divide total train loss by length of train dataloader (average loss per batch per epoch)
      test_loss /= len(data_loader)
      test_acc1 /= len(data_loader)
      test_acc5 /= len(data_loader)
      
      print(f'Test loss: {test_loss:.5f} | Test acc1: {test_acc1:.2f}% | Test acc5: {test_acc5:.2f}%')

   return {"model_name": model.__class__.__name__, 
           "model_loss": test_loss.item(),
           "model_acc1": test_acc1,
           "model_acc5": test_acc5}


def train_step_label(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, 
               accuracy_fn,
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

def test_step_label(model: torch.nn.Module, 
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