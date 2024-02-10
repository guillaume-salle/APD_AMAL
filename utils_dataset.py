import os
from pathlib import Path
import subprocess
import requests
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def get_images_and_labels():
    """
    Reads images and their corresponding labels from a specified dataset directory. If the dataset is not
    present, it attempts to clone a GitHub repository containing the dataset. It processes the images by
    converting them from BGR to RGB format, normalizes the image pixel values to be between 0 and 255, and
    adjusts the labels for zero-based indexing. It also retrieves a dictionary of ImageNet class labels from
    an external URL.

    Note:
    - The function assumes images are stored in PNG format.
    - Two specific images are excluded from processing based on their filenames.
    - Image pixel values are initially normalized to the range [0, 255] as uint8, but it's important to
      normalize them further to [0, 1] as float32 before using them as model input.
    - Labels are adjusted to be zero-based and converted to one-hot encoded format, assuming a total of
      1000 classes for the ImageNet dataset.

    Returns:
        X (torch.Tensor): A tensor of shape (N, H, W, C) containing the RGB images as uint8, where N is the
                          number of images, H and W are the height and width of the images, and C is the
                          number of channels (3 for RGB).
        y (torch.Tensor): A tensor of shape (N,) containing the zero-based integer labels for each image.
        one_hot_y (torch.Tensor): A tensor of shape (N, 1000) containing the one-hot encoded labels for
                                  each image.
        labels_dict (dict): A dictionary mapping integer class IDs to their corresponding human-readable
                            labels as per the ImageNet dataset.
    """
    images_folder_path = Path('tf_to_pytorch_model/dataset/images')
    label_file_path = Path('tf_to_pytorch_model/dataset/dev_dataset.csv')
    
    # Check if the images folder exists; if not, clone the repository
    if not images_folder_path.exists():
        repo_url = 'https://github.com/ylhz/tf_to_pytorch_model.git'
        try:
            subprocess.run(["git", "clone", repo_url], check=True)
            print(f"Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), {}  # Early exit if cloning fails

    labels_df = pd.read_csv(label_file_path)
    labels_dict = pd.Series(labels_df.TrueLabel.values, index=labels_df.ImageId).to_dict()

    images = []
    labels = []

    for filename in os.listdir(images_folder_path):
        if filename.endswith(".png") and filename not in ["362e4ac62cf888f4.png", "bd3617fcc985fe31.png"]:
            img_id = filename[:-4]  # Remove '.png'
            img_path = images_folder_path / filename

            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            images.append(img)

            labels.append(labels_dict.get(img_id, None))

    # Convert the list of images to a single NumPy array first for efficiency
    images_np = np.stack(images).astype(np.uint8)
    X = torch.tensor(images_np, dtype=torch.uint8)  # Convert to tensor
    
    # Ensure labels are integers and convert to a PyTorch tensor
    labels_np = np.array(labels, dtype=np.int32) - 1  # Adjust labels
    y = torch.tensor(labels_np, dtype=torch.long)  # Convert to tensor
    one_hot_y = F.one_hot(y, num_classes=1000).float()

    # Load ImageNet class labels corresponding to numbers
    LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    labels_response = requests.get(LABELS_URL)
    labels_dict = labels_response.json()

    return X, y, one_hot_y, labels_dict
