import matplotlib.pyplot as plt
import numpy as np
import torch
from utils_cam import generate_cams, get_centers


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
    for i in range(min(24, num_images)):
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
        
        label = labels[i].item() if hasattr(labels[i], 'item') else labels[i]  # Handle both tensors and direct values
        label_name = labels_dict.get(str(label), ['Unknown'])[1]
        plt.title(label_name)
        plt.axis('off')

    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
 
def show_cams(model, images, labels, labels_dict):
    """
    Visualizes grayscale class activation maps overlaid on the original images,
    resizing the CAMs to match the original image sizes for proper visualization.
    Adds a title to the figure with the model name and displays actual labels on each image.

    Parameters:
        model (torch.nn.Module): The model used to generate the CAMs, with 'transform' and 'cam' methods.
        images (torch.Tensor): Original images as a batch tensor of shape (N, C, H, W)
                               and dtype torch.uint8. Only the first 24 images will be processed.
        labels (torch.Tensor): The labels corresponding to each image.
        labels_dict (dict): A dictionary mapping label IDs to their actual names.
    """
    images = images[:24]
    labels = labels[:24]

    grayscale_cam = generate_cams(model, images, labels)

    processed_images = []

    for i in tqdm(range(24), desc="Processing Images"):
        img = images[i].float().div(255.0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  
        
        cam_resized = cv2.resize(grayscale_cam[i], (img.shape[1], img.shape[0]))
        processed_images.append(show_cam_on_image(img, cam_resized, use_rgb=True))

    title = f'{model.__class__.__name__} CAMs'
    show_images(processed_images, labels, labels_dict, title=title)
        


def show_centers(model, images, labels, model_name):
    images = images[:24]
    labels = labels[:24]

    grayscale_cam = generate_cams(model, images, labels)

    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    axes = axes.flatten()

    fig.suptitle(f'{model_name} CAMs with Centers', fontsize=16)

    for i, ax in enumerate(axes):
        ax.imshow(grayscale_cam[i], cmap='gray')

        filtered_coordinates = get_centers(grayscale_cam[i], ratio_threshold = 0.6, min_distance = 20)

        # Plot the filtered coordinates on the images
        ax.plot(filtered_coordinates[:, 1], filtered_coordinates[:, 0], 'r.', markersize=10)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()