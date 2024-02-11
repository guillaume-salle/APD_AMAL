from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils_adversarial import get_centers
from utils_plot import show_images


def generate_cams(model, images, labels):
    """
    Generates class activation maps for a set of images and class IDs using a specified model.

    Parameters:
        model (torch.nn.Module): The model to generate CAMs for.
        input_images (torch.Tensor): A batch of input images.
        class_ids (list): A list of class IDs for which to generate CAMs.

    Returns:
        numpy.ndarray: An array of class activation maps for the input images, with
                each CAM resized to match the dimensions of its corresponding input image.
    """
    input_tensor = model.transform(images).detach().requires_grad_(True)
    targets = [ClassifierOutputTarget(class_id) for class_id in labels]
    cam = GradCAMPlusPlus(model=model, target_layers=model.target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam

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
        


# def show_centers(model, images, labels