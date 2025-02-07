import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    best_val_accuracy = 0.0
    last_val_accuracy = 0.0
    weights = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 50 == 0:
                print(f"Progress of epoch {epoch + 1}/{epochs}: loop {i}/{len(train_loader)} finished")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        eval_data = evaluate_model(model, val_loader, criterion, device)  # Validation accuracy
        val_accuracy = eval_data['accuracy']

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {epoch_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

        # Store training history
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)
        history["val_accuracy"].append(val_accuracy)

        # Track best validation accuracy
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        last_val_accuracy = val_accuracy  # Store last epoch validation accuracy

    return last_val_accuracy, best_val_accuracy, history


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader and compute metrics.
    """

    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []
    running_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()

            # Convert outputs to binary predictions (0 or 1)
            predictions = (outputs >= 0.5).int().squeeze()

            # Store true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Compute average loss
    avg_loss = running_loss / len(dataloader)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Compute EER
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    # Return metrics as a dictionary
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer": eer,
    }


def get_best_hyperparameters(criterion, train_dataset, val_dataset, model_constructor):
    # Define hyperparameter grid
    param_grid = {
        'batch_size': [64, 32],
        'learning_rate': [0.005, 0.01],
        'epochs': [7],
        'optimizer': ['Adam', 'RMSprop']
    }

    # Map optimizer names to their corresponding PyTorch classes
    optimizer_map = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop
    }

    # Device setup
    device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Chosen device: {device}")

    # Perform grid search
    best_accuracy = 0.0
    best_params = None
    best_history = None

    for params in ParameterGrid(param_grid):
        print(f"Testing hyperparameters: {params}")

        # Create DataLoader with current batch size
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=4)

        model = model_constructor(device)

        # Move model to GPU if available
        device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Initialize optimizer based on the current grid search parameters
        optimizer_class = optimizer_map[params['optimizer']]
        optimizer = optimizer_class(model.parameters(), lr=params['learning_rate'])

        # Train the model
        last_val_accuracy, best_val, history = train_model(model, train_loader, val_loader, criterion,
                                                                       optimizer, params['epochs'], device)
        print(f"Best model validation score during training: {best_val}")

        # Track the best hyperparameters
        if last_val_accuracy > best_accuracy:
            best_accuracy = last_val_accuracy
            best_params = params
            best_history = history

        print(f"Validation Accuracy: {last_val_accuracy:.2f}%")
        print("----------------------------------------")

    # Print the best hyperparameters
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    return best_params.copy(), best_accuracy, best_history