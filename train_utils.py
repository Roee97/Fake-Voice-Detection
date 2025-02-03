import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    best_val_accuracy = 0.0
    last_val_accuracy = 0.0

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
        val_accuracy = evaluate_model(model, val_loader, device)  # Validation accuracy

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


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.to(torch.float32)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    return val_accuracy
