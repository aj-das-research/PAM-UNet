import torch
import matplotlib.pyplot as plt
from model import PAMUNet
from data_loading import get_dataloader
from torchvision import transforms

def visualize_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    
    inverse_transform = transforms.Compose([
        transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]),
    ])

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            img = inverse_transform(images[0]).cpu().permute(1, 2, 0).numpy()
            axes[0].imshow(img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            # Ground truth mask
            axes[1].imshow(masks[0].cpu().numpy(), cmap='gray')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')

            # Predicted mask
            axes[2].imshow(preds[0].cpu().numpy(), cmap='gray')
            axes[2].set_title("Prediction")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(f"prediction_{i+1}.png")
            plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_classes = 2  # Change this based on your dataset
    batch_size = 8

    # Data loader
    test_loader = get_dataloader('path/to/test/images', 'path/to/test/masks', batch_size, 'test')

    # Model
    model = PAMUNet(in_channels=3, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    # Visualize predictions
    visualize_predictions(model, test_loader, device)