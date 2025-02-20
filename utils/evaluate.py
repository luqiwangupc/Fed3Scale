import torch
from tqdm import tqdm


def evaluate(encoded_model, model, val_loader, criterion, device, process=False):
    encoded_model.to(device)
    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader) if process else val_loader:
            inputs, labels = batch['img'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)

            encoded_inputs = encoded_model(inputs)

            outputs = model(encoded_inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)

    return {'val_loss': avg_loss, 'val_accuracy': accuracy}
