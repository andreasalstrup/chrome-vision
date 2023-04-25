import torch
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score
from pathlib import Path

def saveModel(path, name, model: nn.Module):
    MODEL_PATH = Path(path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f'Model saved to: {MODEL_SAVE_PATH}')
    torch.save(obj=model.state_dict(),
            f=MODEL_SAVE_PATH)

def loadModel(path, model: nn.Module):
    loaded_model = model
    loaded_model.load_state_dict(torch.load(f=path))
    print(f'loaded model: {loaded_model.state_dict()}')

def saveCheckpoint(path, model, optimizer, epoch, showTraining, showTesting, train_data_loader, test_data_loader):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'ShowTraining': showTraining,
        'ShowTesting': showTesting,
        'train_data_loader': train_data_loader,
        'test_data_loader': test_data_loader
    }, path)

def loadCheckpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    showTraining = checkpoint['ShowTraining']
    showTesting = checkpoint['ShowTesting']
    train_data_loader = checkpoint['train_data_loader']
    test_data_loader = checkpoint['test_data_loader']
    return model, optimizer, epoch, showTraining, showTesting, train_data_loader, test_data_loader

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Delete
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# def accuracy_top_k(y_true, y_pred, k=1):
#     with torch.inference_mode():
#         top_k_preds = [np.argpartition(pred, max(-k, 1))[-k:] for pred in y_pred.cpu()]
#         y_pred_top_k = [pred in top_k for pred, top_k in zip(y_pred.cpu(), top_k_preds)]
#         return accuracy_score(y_true, y_pred_top_k)
    
def accuracy_top_k(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) # view() -> reshape()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res