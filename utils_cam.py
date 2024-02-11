from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
import torch


def generate_cams(model, input_images, class_ids):
    """
    Generates class activation maps for a set of images and class IDs using a specified model.
    The device is inputed from the model device.

    Parameters:
        model (torch.nn.Module): The model to generate CAMs for.
        input_images (torch.Tensor): A batch of input images.
        class_ids (list): A list of class IDs for which to generate CAMs.

    Returns:
        numpy.ndarray: An array of class activation maps for the input images, with
                each CAM resized to match the dimensions of its corresponding input image.
    """
    device = model.parameters().__next__().device
    input_tensor = model.transform(input_images).detach().requires_grad_(True)
    input_tensor = input_tensor.to(device)
    
    targets = [ClassifierOutputTarget(class_id) for class_id in class_ids]
    targets = targets.to(device)
    
    cam = GradCAMPlusPlus(model=model, target_layers=model.target_layers)
    print(cam.device)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam

def print_cams(model_name, grayscale_cams, images, labels, labels_dict):
    """
    Visualizes grayscale class activation maps overlaid on the original images,
    resizing the CAMs to match the original image sizes for proper visualization.
    Adds a title to the figure with the model name and displays actual labels on each image.

    Parameters:
        grayscale_cams (numpy.ndarray): An array of grayscale class activation maps.
        images (torch.Tensor): Original images as a batch tensor of shape (N, C, H, W)
                            and dtype torch.uint8.
        model_name (str): The name of the model used to generate the CAMs.
        labels (list or torch.Tensor): The labels corresponding to each image.
        labels_dict (dict): A dictionary mapping label IDs to their actual names.
    """
    num_images_to_display = min(25, len(images))
    columns = 5
    rows = math.ceil(num_images_to_display / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i, cam_image in enumerate(grayscale_cams):
        if i >= num_images_to_display:
            break
        img = images[i].float().div(255.0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert to HWC for visualization
        
        # Resize grayscale CAM to match the original image size
        cam_resized = cv2.resize(cam_image, (img.shape[1], img.shape[0]))
        visualization = show_cam_on_image(img, cam_resized, use_rgb=True, image_weight=0.7)
        
        axes[i].imshow(visualization)
        axes[i].axis('off')

        # Get the actual label name using labels_dict
        label_id = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
        label_name = labels_dict[str(label_id)][1]

        # Display the actual label name for each image
        axes[i].set_title(label_name, fontsize=12, pad=0)

    # Set the title of the entire figure to include the model name
    fig.suptitle(f"Class Activation Maps for {model_name}", fontsize=16)
    plt.subplots_adjust(top=0.96, hspace=0.0, wspace=0.1)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the title
    plt.show()
