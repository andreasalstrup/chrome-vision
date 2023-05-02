import torch
from torch import device, nn
from tqdm.auto import tqdm
import math


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
      query_image = images[0].to(device) ### Are there only 2 elements in images?
      key_image = images[1].to(device)

      # 1. Forward pass
      output, target = model(query_batch_images=query_image,key_batch_images=key_image)
      #output.requires_grad = True

      # 2. Calculate loss (per batch)
      loss = loss_fn(output, target)
      train_loss += loss.item()

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
   return train_loss, train_acc1, train_acc5


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
         query_image = images[0].to(device)
         key_image = images[1].to(device)

         # Make predictions
         output, target = model(query_batch_images=query_image,key_batch_images=key_image)

         # Calculate loss 
         loss = loss_fn(output, target)
         test_loss += loss.item()

         # Calculate accuracy
         acc1, acc5 = accuracy_fn(output, target, topk=(1, 5))
         test_acc1 += acc1.item()
         test_acc5 += acc5.item()

      # Divide total train loss by length of train dataloader (average loss per batch per epoch)
      test_loss /= len(data_loader)
      test_acc1 /= len(data_loader)
      test_acc5 /= len(data_loader)
      
      print(f'Test loss: {test_loss:.5f} | Test acc1: {test_acc1:.2f}% | Test acc5: {test_acc5:.2f}%')

   return test_loss, test_acc1, test_acc5


def eval_model(name: str, 
               model: torch.nn.Module, 
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
         query_image = images[0].to(device)
         key_image = images[1].to(device)

         # Make predictions
         output, target = model(query_batch_images=query_image,key_batch_images=key_image)

         # Calculate loss 
         loss = loss_fn(output, target)
         test_loss += loss.item()

         # Calculate accuracy
         acc1, acc5 = accuracy_fn(output, target, topk=(1, 5))
         test_acc1 += acc1.item()
         test_acc5 += acc5.item()

      # Divide total train loss by length of train dataloader (average loss per batch per epoch)
      test_loss /= len(data_loader)
      test_acc1 /= len(data_loader)
      test_acc5 /= len(data_loader)

   return {"model_name": name, 
           "model_loss": test_loss,
           "model_acc1": test_acc1,
           "model_acc5": test_acc5}


def adjust_learning_rate(optimizer, epoch, epochs, lr):
   optimizer.param_groups[0]['lr'] *= 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))     
   return optimizer