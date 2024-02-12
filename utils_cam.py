from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from skimage.feature import peak_local_max
import numpy as np

def generate_cams(model, images, labels, method="++"):
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
    if method == "++":
        cam = GradCAMPlusPlus(model=model, target_layers=model.target_layers)
    elif method == "normal":
        cam = GradCAM(model=model, target_layers=model.target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam


def get_centers(grayscale_cam, ratio_threshold=0.6, min_distance=20, num_peaks=3):
    """
    Identifies and filters the coordinates of local maxima in a grayscale class activation map (CAM)
    based on specified intensity ratio threshold, minimum distance between maxima, and number of peaks.

    Parameters:
    - grayscale_cam (ndarray): A 2D NumPy array representing a grayscale CAM.
    - ratio_threshold (float, optional): Minimum ratio of local maximum's intensity to the highest maximum's intensity.
    - min_distance (int, optional): Minimum number of pixels separating local maxima.
    - num_peaks (int, optional): Maximum number of peaks to identify.

    Returns:
    - filtered_coordinates (ndarray): Array of filtered coordinates of local maxima.
    """
    coordinates = peak_local_max(grayscale_cam, num_peaks=num_peaks, exclude_border=False, min_distance=min_distance)

    if len(coordinates) == 0:
        return np.array([np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)])

    maxima_values = grayscale_cam[coordinates[:, 0], coordinates[:, 1]]
    highest_maxima_value = max(maxima_values)
    return coordinates[maxima_values >= ratio_threshold * highest_maxima_value]
