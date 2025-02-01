"""
Goal: Train and evaluate the final model

This script handles the training of the final model using previously
determined hyper parameters from train.ipynb.
"""

# Dependencies
# ------------

# standard library imports
import logging
from pathlib import Path
from PIL import Image
from statistics import mean, stdev

# third-party library imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


# Constants and configurations
# ----------------------------

# set up device based on available hardware
device = (
    "mps" if torch.backends.mps.is_available() # for Apple Silicon chips
    else "cuda" if torch.cuda.is_available() # for NVIDIA GPUs
    else "cpu" # for CPU
)

# paths
PATH_REPO = Path(__file__).parent.parent
PATH_DATA = PATH_REPO / "data/processed"
PATH_TRAIN = PATH_DATA / "train"
PATH_TEST = PATH_DATA / "test"
PATH_MODELS = PATH_REPO / "models"
PATH_FINAL_MODEL = PATH_MODELS / "final_model.pth"

# training hyper parameters
BATCH_SIZE = 32
N_EPOCHS = 2 # FIXME: change back to 50 - 2 is used for prototyping
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 0.001

# image transformations
TRANSFORM = transforms.Compose([
    # convert PIL images to PyTorch tensor and scale pixel value range to 0-1
    transforms.ToTensor(),
    # normalize image image according to ImageNet statistics
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Classes
# -------

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self,
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        path=PATH_MODELS / "final_model_hpo_checkpoint.pt"
    ):
        """
        Args:
            patience (int): How many epochs to wait before stopping when loss is
            not improving
            min_delta (float): Minimum change in the monitored quantity to 
            qualify as an improvement
            path (str): Path for the checkpoint to be saved to
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        self.best_model = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """
        Check if validation loss has improved and save model if it has.
        
        Args:
            val_loss (float): validation loss
            model (torch.nn.Module): the model to save
            
        Returns:
            None
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """
        Saves model when validation loss decreases.
        
        Args:
            model (torch.nn.Module): the model to save
        
        Returns:
            None
        """
        torch.save(model.state_dict(), self.path)
        self.best_model = model.state_dict()


# Functions
# ---------

def custom_loader(path: str) -> Image.Image:
    """
    Custom image loader that forces image reading in RGB mode.
    If image cannot be loaded, return a default image with mean values
    matching ImageNet statistics.
    
    Args:
        path (str): path to the image file
    
    Returns:
        PIL.Image.Image: loaded image in RGB mode or default image
    """
    try:
        # force RGB mode during initial open
        img = Image.open(path).convert("RGB")
        
        return img
    
    except Exception as e:
        print(f"Error loading image {path}: {str(e)}")
    
        # return a default image as fallback with mean values matching ImageNet
        return Image.new("RGB", (224, 224), (128, 128, 128))


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> tuple:
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): the model to train
        train_loader (torch.utils.data.DataLoader): the training data loader
        criterion (torch.nn.Module): the loss function
        optimizer (torch.optim.Optimizer): the optimizer
        device (str): the device to use for training
    
    Returns:
        tuple: average loss and accuracy for the epoch
    """
    
    # set model to training mode
    model.train()
    # initialize running loss and accuracy
    running_loss = 0.0 # accumulator for loss across batches
    correct = 0 # accumulator for correct predictions across batches
    total = 0 # accumulator for total predictions across batches

    # iterate over the batches of training data
    for inputs, labels in train_loader:
        # move each batch of data to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero / reset the parameter gradients from previous bachward passes
        optimizer.zero_grad()
        
        # forward pass
        # get model predictions
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, labels)
        
        # backward pass
        # compute gradients
        loss.backward()
        # update model parameters / weights
        optimizer.step()
        
        # accumulate loss and statistics
        # accumulate batch loss
        running_loss += loss.item()
        # get predicted class
        _, predicted = outputs.max(1)
        # accumulate total predictions
        total += labels.size(0)
        # accumulate correct predictions
        correct += predicted.eq(labels).sum().item()
    
    # calculate epoch statistics
    # average loss over all batches
    epoch_loss = running_loss / len(train_loader)
    # calculate accuracy
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str
) -> dict:
    """
    Validate the model.
    
    Args:
        model (torch.nn.Module): the model to validate
        val_loader (torch.utils.data.DataLoader): the validation data loader
        criterion (torch.nn.Module): the loss function
        device (str): the device to use for validation
    
    Returns:
        dict: validation metrics
    """
    
    # set model to evaluation mode
    model.eval()
    # initialize running loss and accuracy
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    # iterate over the batches of validation data
    with torch.no_grad():
        
        # iterate over the batches of validation data
        for inputs, labels in val_loader:
            
            # move each batch of data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # get model predictions
            outputs = model(inputs)
            # compute loss
            loss = criterion(outputs, labels)
            
            # accumulate loss
            val_loss += loss.item()
            
            # get predictions and probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # collect all predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            
    # calculate metrics
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * accuracy_score(all_labels, all_preds)
    val_roc_auc = roc_auc_score(all_labels, all_probs)
    val_f1 = f1_score(all_labels, all_preds)
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_roc_auc': val_roc_auc,
        'val_f1': val_f1
    }


def evaluate_model(
    model_name: str,
    validation_results: list,
    include_model_info: bool = False
) -> dict:
    """
    Summarize model performance across validation results.
    
    Args:
        model_name (str): name of the model for identification
        history (dict): dictionary containing training history
            each dictionary should have 'train_loss', 'train_acc', 'val_metrics'
    
    Returns:
        dict: summary statistics for each metric (mean ± std)
    """
    # collect metrics
    metrics = {
        'val_loss': [],
        'val_acc': [],
        'val_roc_auc': [],
        'val_f1': []
    }
    
    # collect metrics from all validation results
    for result in validation_results:
        metrics['val_loss'].append(result['val_loss'])
        metrics['val_acc'].append(result['val_acc'])
        metrics['val_roc_auc'].append(result['val_roc_auc'])
        metrics['val_f1'].append(result['val_f1'])
    
    # initialize summary dictionary
    summary = {}
    
    # include info about model if desired
    if include_model_info:
        summary['model'] = model_name
        
    # compute mean and standard deviation and add to summary
    summary['val_loss'] = f"{mean(metrics['val_loss']):.4f} ± {stdev(metrics['val_loss']):.4f}"
    summary['val_acc'] = f"{mean(metrics['val_acc']):.2f}% ± {stdev(metrics['val_acc']):.2f}%"
    summary['val_roc_auc'] = f"{mean(metrics['val_roc_auc']):.4f} ± {stdev(metrics['val_roc_auc']):.4f}"
    summary['val_f1'] = f"{mean(metrics['val_f1']):.4f} ± {stdev(metrics['val_f1']):.4f}"
    
    return summary


# Main
# ----

def main():
    
    
    # print which device is being used
    if device == "cpu":
        print("No GPU available. Training on CPU.")
    elif device == "mps" or device == "cuda":
        print(f"{device.capitalize()} available. Training on GPU.")


    # recreate datasets with new loader and transforms
    train_dataset = datasets.ImageFolder(
        PATH_TRAIN,
        loader=custom_loader,
        transform=TRANSFORM
    )
    test_dataset = datasets.ImageFolder(
        PATH_TEST,
        loader=custom_loader,
        transform=TRANSFORM
    )

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # print info about training data
    print(f"number of training images: {len(train_dataset)}")
    print(f"number of test images: {len(test_dataset)}")
    print(f"classes: {train_dataset.classes}")

    # initialize model
    model = models.resnet18(weights="IMAGENET1K_V1")

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
        
    # modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # move model to GPU
    model = model.to(device)

    # setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    early_stopping_baseline = EarlyStopping(
        patience=7,
        min_delta=0.001,
        path=PATH_MODELS / "model_checkpoint_baseline.pt"
    )

    # training loop
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}

    for epoch in range(N_EPOCHS):
        
        # train for one epoch
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        
        # validate 
        val_metrics = validate(
            model,
            test_loader,
            criterion,
            device
        )
        
        # store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_acc']:.2f}%"
        )
        print("-" * 50)
        
        # check early stopping
        early_stopping_baseline(val_metrics['val_loss'], model)
        if early_stopping_baseline.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            model.load_state_dict(early_stopping_baseline.best_model)
            break

    # set the model to evaluation mode
    model.eval()

    # evaluate model on test set
    test_metrics = validate(
        model,
        val_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # print results in a formatted way
    print("\nFinal Test Set Metrics:")
    print("-" * 50)
    print(f"Test Loss:     {test_metrics['val_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['val_acc']:.2f}%")
    print(f"Test ROC AUC:  {test_metrics['val_roc_auc']:.4f}")
    print(f"Test F1 Score: {test_metrics['val_f1']:.4f}")


# Main
# ----
if __name__ == "__main__":
    main()

