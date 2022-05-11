import torch
import torch.nn as nn

def accuracy(dataloader, model):
    correct = 0
    total = 0
    running_loss = 0
    n = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval() # Dropout 비활성화.
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _ = model(images) # x, att_list
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
            running_loss += loss.item()

        loss_result = running_loss / n

    acc = 100 * correct / total
    model.train()
    return acc, loss_result