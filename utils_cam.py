from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import math


def generate_cams(model, input_images, class_ids, device='cpu'):
    """
    Generates class activation maps for a set of images and class IDs using a specified model.

    Parameters:
        model (torch.nn.Module): The model to generate CAMs for.
        input_images (torch.Tensor): A batch of input images.
        class_ids (list): A list of class IDs for which to generate CAMs.
        device (str): The device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor containing the grayscale class activation maps.
    """
    input_tensor = model.transform(input_images).to(device).detach().requires_grad_(True)
    
    targets = [ClassifierOutputTarget(class_id) for class_id in class_ids]
    
    cam = GradCAMPlusPlus(model=model, target_layers=model.target_layers)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam

def print_cams(grayscale_cams, images):
    """
    Visualizes grayscale class activation maps overlaid on the original images.

    Parameters:
        grayscale_cams (torch.Tensor): The grayscale class activation maps.
        images (torch.Tensor): The original images.
    """
    num_images_to_display = min(25, len(images))
    columns = 5
    rows = math.ceil(num_images_to_display / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i, cam_image in enumerate(grayscale_cams):
        if i >= num_images_to_display:
            break
        img = images[i].float() / 255.0 
        img = img.permute(1, 2, 0).cpu().numpy()
        visualization = show_cam_on_image(img, cam_image, use_rgb=True)
        axes[i].imshow(visualization)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
