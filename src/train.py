import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import PAMUNet
from data_loading import get_dataloader

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_classes = 2  
    learning_rate = 0.001
    num_epochs = 200
    batch_size = 8

    # Data loaders
    train_loader = get_dataloader('path/to/train/images', 'path/to/train/masks', batch_size, 'train')
    val_loader = get_dataloader('path/to/val/images', 'path/to/val/masks', batch_size, 'val')

    # Model, loss, and optimizer
    model = PAMUNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)