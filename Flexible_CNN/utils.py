import random
import numpy as np
import torch
import tqdm
import torch.optim as optim
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate(model, val_loader, criterion, device):
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return val_loss / total, correct / total

def train_model(model, train_loader, val_loader, device,learning_rate=0.1, epochs=5,wandb=None ):
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

        for epoch in tqdm(range(epochs), desc="Training Progress",position=0):
        # for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",position=1):
            # for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                # print(images.shape, labels.shape)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc = validate(model , val_loader, criterion, device)

            if wandb:
                wandb.log({"train_loss": train_loss, "train_acc": train_acc,"val_loss": val_loss, "val_acc": val_acc})
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f},train_loss: {train_loss:.4f},val_loss: {val_loss:.4f}")