import torch
from torch import nn
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