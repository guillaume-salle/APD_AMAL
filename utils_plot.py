import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(images, labels, labels_dict, title=None):
    """
    Displays a grid of images with their corresponding labels.

    Handles both PyTorch tensors and lists of images (NumPy arrays or PIL images),
    automatically adjusting for the input format. Displays up to 24 images in a 4x6 grid.
    Images are expected to be in CHW format if PyTorch tensors, or HWC format if NumPy arrays.

    Parameters:
        images (torch.Tensor or list): A batch of images, either as a PyTorch tensor
                                       with shape (N, C, H, W) or a list of NumPy arrays/PIL images.
        labels (iterable): An iterable of labels corresponding to each image.
        labels_dict (dict): A dictionary mapping numerical labels to their string representations.
        title (str, optional): Title for the entire figure. Defaults to None.

    Returns:
        None. Displays a matplotlib figure with the images and titles.
    """
    # Determine the number of images
    num_images = len(images) if isinstance(images, list) else images.shape[0]

    plt.figure(figsize=(15, 12))
    for i in range(min(24, num_images)):  # Process up to 24 images or the total number of images if fewer
        plt.subplot(4, 6, i + 1)
        
        # Handle images if they are in a list or a PyTorch tensor
        if isinstance(images, list):
            img = images[i]
            # If images are NumPy arrays, they might already be in HWC format
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                pass  # NumPy images are likely already in HWC format
            elif isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()  # Convert PyTorch tensor to HWC for visualization
        else:  # images is a PyTorch tensor
            img = images[i].permute(1, 2, 0).numpy()

        plt.imshow(img)
        
        # Retrieve and display the label
        label = labels[i].item() if hasattr(labels[i], 'item') else labels[i]  # Handle both tensors and direct values
        label_name = labels_dict.get(str(label), ['Unknown'])[1]  # Safely get the label name with a fallback
        plt.title(label_name)
        plt.axis('off')

    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()  # Use plt.tight_layout() for better spacing
    plt.show()
 