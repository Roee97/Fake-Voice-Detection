import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            labels = labels.float()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 50 == 0:
                print(f"Progress - enumeration {i} / {len(train_loader)}")

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        # Evaluate on validation set
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%"
              )

        # Track the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    return best_val_accuracy


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    return val_accuracy
