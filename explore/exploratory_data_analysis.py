import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split

def load_and_visualize(image_path, mask_path):
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Mask')
    ax2.axis('off')
    
    plt.savefig('sample_image_and_mask.png')
    plt.close()

def get_dataset_stats(image_dir, mask_dir):
    image_sizes = []
    mask_ratios = []
    
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '_mask.png'))
        
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        image_sizes.append(img.shape[:2])
        mask_ratios.append(np.sum(mask > 0) / mask.size)
    
    return image_sizes, mask_ratios

def plot_dataset_stats(image_sizes, mask_ratios):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(*zip(*image_sizes))
    plt.title('Image Sizes')
    plt.xlabel('Width')
    plt.ylabel('Height')

    plt.subplot(122)
    sns.histplot(mask_ratios, kde=True)
    plt.title('Mask Ratio Distribution')
    plt.xlabel('Mask Ratio')
    plt.savefig('dataset_stats.png')
    plt.close()

def get_class_distribution(mask_dir):
    class_distribution = []
    
    for mask_name in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_name)
        mask = np.array(Image.open(mask_path))
        unique, counts = np.unique(mask, return_counts=True)
        class_distribution.append(dict(zip(unique, counts)))
    
    return class_distribution

def plot_class_distribution(class_distribution):
    class_counts = {0: 0, 1: 0}  # Assuming binary segmentation
    for d in class_distribution:
        for k, v in d.items():
            class_counts[k] += v

    plt.figure(figsize=(8, 6))
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title('Overall Class Distribution')
    plt.savefig('class_distribution.png')
    plt.close()

def get_intensity_distribution(image_dir):
    intensities = []
    
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale
        intensities.extend(img.flatten())
    
    return intensities

def plot_intensity_distribution(intensities):
    plt.figure(figsize=(10, 6))
    sns.histplot(intensities, kde=True, bins=50)
    plt.title('Image Intensity Distribution')
    plt.xlabel('Intensity')
    plt.savefig('intensity_distribution.png')
    plt.close()

def main():
    # Set paths
    image_dir = 'data/train/images'
    mask_dir = 'data/train/masks'

    # 1. Load and visualize sample image
    sample_image_path = os.path.join(image_dir, os.listdir(image_dir)[0])
    sample_mask_path = os.path.join(mask_dir, os.listdir(mask_dir)[0])
    load_and_visualize(sample_image_path, sample_mask_path)
    print("Sample image and mask visualization saved.")

    # 2. Analyze dataset statistics
    image_sizes, mask_ratios = get_dataset_stats(image_dir, mask_dir)
    plot_dataset_stats(image_sizes, mask_ratios)
    print("Dataset statistics plotted and saved.")

    # 3. Analyze class distribution
    class_distribution = get_class_distribution(mask_dir)
    plot_class_distribution(class_distribution)
    print("Class distribution plotted and saved.")

    # 4. Analyze image intensity distribution
    intensities = get_intensity_distribution(image_dir)
    plot_intensity_distribution(intensities)
    print("Image intensity distribution plotted and saved.")

    # 5. Print conclusions
    print("\nConclusions:")
    print("1. The dataset contains images of varying sizes, which may require resizing or padding during preprocessing.")
    print("2. There is a class imbalance in the segmentation masks, which might need to be addressed in the loss function or sampling strategy.")
    print("3. The image intensity distribution shows... (interpret the results)")

    print("\nNext steps:")
    print("1. Implement data augmentation techniques to increase dataset diversity")
    print("2. Consider using a weighted loss function to address class imbalance")
    print("3. Normalize image intensities during preprocessing")
    print("4. Experiment with different input sizes for the PAM-UNet model")

if __name__ == "__main__":
    main()