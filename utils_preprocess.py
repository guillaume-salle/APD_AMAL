import numpy as np
import skimage.transform
import torch
import torch.nn.functional as F

mean_imagenet = [0.485, 0.456, 0.406]
std_imagenet = [0.229, 0.224, 0.225]

def resize_image(input_images, model_name):
    """
    Resizes images to the required input dimensions of specified models using PyTorch tensors.

    Parameters:
        input_images (torch.Tensor): Input image or batch of images. Expected to be 3-dimensional (C, H, W) for a single image or 4-dimensional (N, C, H, W) for a batch.
        model_name (str): Name of the model to resize images for. Supported models: 'squeezenet', 'inceptionv4', 'resnet50', 'adv_inceptionv3'.

    Returns:
        torch.Tensor: Resized image or batch of images with dimensions required by the specified model.

    Raises:
        ValueError: If `model_name` is not recognized or if `input_images` does not have 3 or 4 dimensions.
    """
    model_shapes = {
        "squeezenet": (224, 224),
        "inceptionv4": (299, 299),
        "resnet50": (224, 224),
        "adv_inceptionv3": (299, 299)
    }

    target_shape = model_shapes.get(model_name)
    if target_shape is None:
        raise ValueError("Invalid model name. Please provide a valid model name.")
    
    # Ensure the input is a float tensor for interpolation
    if not input_images.is_floating_point():
        input_images = input_images.float()

    if len(input_images.shape) == 4:  # Batch of images, shape: (N, C, H, W)
        size = target_shape
    elif len(input_images.shape) == 3:  # Single image, shape: (C, H, W)
        size = target_shape
        input_images = input_images.unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError("Input images should have 3 or 4 dimensions.")

    # Resize image
    resized_images = F.interpolate(input_images, size=size, mode='bilinear', align_corners=False)
    
    if input_images.shape[0] == 1:  # If it was a single image, remove the batch dimension
        resized_images = resized_images.squeeze(0)

    return resized_images

"""
class Preprocessing_Transform:
    def __init__(self, model_name):
        self.mean = mean_imagenet
        self.std = std_imagenet
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, img):
        img = resize_image(img, self.model_name)
        if len(img.shape)==3:
          img_normalized = (img - np.array(self.mean)[None, None, :]) / np.array(self.std)[None, None, :] # Normalize
          return torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1).to(self.device)
        elif len(img.shape)==4:
          img_normalized = (img - np.array(self.mean)[None, None, None, :]) / np.array(self.std)[None, None, None, :] # Normalize
          return torch.tensor(img_normalized, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

class Depreprocessing_Transform:
    def __init__(self, mean, std, model_name):
        self.mean = mean
        self.std = std
        self.model_name = model_name

    def __call__(self, tensor_img):
        if len(tensor_img.shape)==3:
          img_normalized = tensor_img.permute(1,2,0).detach().cpu().numpy()
          img = (img_normalized * np.array(self.std)[None, None, :]) + np.array(self.mean)[None, None, :] # Denormalize
          img = resize_image(img, self.model_name)
          return img
        elif len(tensor_img.shape)==4:
          img_normalized = tensor_img.permute(0, 2, 3, 1).detach().cpu().numpy()
          img = (img_normalized * np.array(self.std)[None, None, None, :]) + np.array(self.mean)[None, None, None, :] # Denormalize
          img = resize_image(img, self.model_name)
          return img

class Normalized_Clamp:
    def __init__(self, mean, std, device):
        self.mean = torch.Tensor(mean).to(device)
        self.std = torch.Tensor(std).to(device)

    def __call__(self, normalized_tensor):
        low = (-self.mean/self.std).view(3,1,1).expand_as(normalized_tensor)
        high = ((1-self.mean)/self.std).view(3,1,1).expand_as(normalized_tensor)
        return torch.clamp(normalized_tensor, low, high)
"""