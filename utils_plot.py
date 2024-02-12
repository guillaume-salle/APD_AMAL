import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from tqdm import tqdm
from utils_cam import generate_cams, get_centers
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt
import numpy as np
import torch

def show_images(images, labels, labels_dict, title=None, rows=2):
    """
    Displays a grid of images with their corresponding labels, organizing the images
    into the specified number of rows with 6 columns each.

    Parameters:
        images (torch.Tensor or list): A batch of images, either as a PyTorch tensor
                                       with shape (N, C, H, W) or a list of NumPy arrays/PIL images.
        labels (iterable): An iterable of labels corresponding to each image.
        labels_dict (dict): A dictionary mapping numerical labels to their string representations.
        title (str, optional): Title for the entire figure. Defaults to None.
        rows (int): The number of rows to organize the images into.

    Returns:
        None. Displays a matplotlib figure with the images and titles.
    
    Raises:
        ValueError: If there are not enough images to fill the specified number of rows and columns.
    """
    num_images = len(images) if isinstance(images, list) else images.shape[0]
    columns = 6  # Fixed number of columns
    total_cells = rows * columns

    # Check if there are enough images to fill the grid
    if num_images < total_cells:
        raise ValueError(f"Not enough images to fill {rows} rows of {columns} columns. Only {num_images} images provided.")

    plt.figure(figsize=(2 * columns, 2.5 * rows))  # Adjust figure size dynamically
    for i in range(total_cells):
        plt.subplot(rows, columns, i + 1)
        
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # Convert PyTorch tensor to HWC format for visualization
        
        plt.imshow(img)
        label = labels[i].item() if hasattr(labels[i], 'item') else labels[i]
        label_name = labels_dict.get(str(label), 'Unknown')[1]
        plt.title(label_name, fontsize=10)
        plt.axis('off')

    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def show_one_image(image, title=None):
    """
    Displays a single image with its corresponding label name.

    Parameters:
        image (torch.Tensor or numpy.ndarray or PIL.Image): The image to be displayed. 
                                                            Should be in CxHxW format if a PyTorch tensor,
                                                            HxWxC if a numpy array.
        title (str, optional): Title for the image. Defaults to None.

    Returns:
        None. Displays the image with matplotlib.
    """
    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to numpy array in HWC format for visualization
        image = image.permute(1, 2, 0).cpu().numpy()


    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    if title:
        plt.title(f"{title} - {label_name}")
    plt.axis('off')
    plt.show()


def show_cams(model, images, labels, labels_dict, rows=2, image_weight=0.7):
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
    num_images = 6 * rows
    images = images[:num_images]
    labels = labels[:num_images]

    grayscale_cam = generate_cams(model, images, labels)

    processed_images = []

    for i in tqdm(range(num_images), desc=f"Processing CAMs for {model.name}"):
        img = images[i].float().div(255.0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  
        
        cam_resized = cv2.resize(grayscale_cam[i], (img.shape[1], img.shape[0]))
        processed_images.append(show_cam_on_image(img, cam_resized, use_rgb=True, image_weight=image_weight))

    title = f'{model.__class__.__name__} CAMs'
    show_images(processed_images, labels, labels_dict, rows=rows, title=title)
        

def show_one_cam(model, img, label, title=None, image_weight=0.7):
    grayscale_cam = generate_cams(model, img.unsqueeze(0), label.unsqueeze(0)).squeeze()

    img = img.float().div(255.0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  
        
    cam_resized = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    processed_img = show_cam_on_image(img, cam_resized, use_rgb=True, image_weight=image_weight)

    show_one_image(processed_img, title=title)


def show_centers(model, images, labels, rows=2):
    """
    Displays a grid of CAMs overlaid with center points.

    Parameters:
        model: The model used for generating CAMs.
        images: The input images.
        labels: The corresponding labels.
        rows: The number of rows to display.

    Returns:
        None. Displays a matplotlib figure with the CAMs and center points.
    """
    num_images = len(images) if isinstance(images, list) else images.shape[0]
    columns = 6  # Fixed number of columns
    total_cells = rows * columns
    if num_images < total_cells:
        raise ValueError(f"Not enough images to fill {rows} rows of {columns} columns. Only {num_images} images provided.")

    images = images[:total_cells]  
    labels = labels[:total_cells]

    grayscale_cam = generate_cams(model, images, labels)

    plt.figure(figsize=(2 * columns, 2.5 * rows))  # Adjust figure size dynamically
    for i in range(total_cells):
        plt.subplot(rows, columns, i + 1)
        
        ax = plt.gca()
        ax.imshow(grayscale_cam[i], cmap='gray')

        filtered_coordinates = get_centers(grayscale_cam[i], ratio_threshold=0.6, min_distance=20)

        # Plot the filtered coordinates on the images
        ax.plot(filtered_coordinates[:, 1], filtered_coordinates[:, 0], 'r.', markersize=10)
        ax.set_axis_off()

    plt.suptitle(f'{model.name} CAMs with Centers', fontsize=16)
    plt.tight_layout()
    plt.show()