import torch
import torch.nn as nn
from tqdm import tqdm
from model import PAMUNet
from data_loading import get_dataloader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    iou = jaccard_score(all_targets, all_preds, average='weighted')

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU Score: {iou:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_classes = 2  
    batch_size = 8

    # Data loader
    test_loader = get_dataloader('path/to/test/images', 'path/to/test/masks', batch_size, 'test')

    # Model and loss
    model = PAMUNet(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    evaluate(model, test_loader, criterion, device)