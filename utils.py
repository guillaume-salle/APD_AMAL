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
    Reads images and labels from a dataset directory, clones a repo if necessary, and
    provides one-hot encoded labels.

    Returns:
        X (numpy.ndarray): Array of RGB images.
        y (numpy.ndarray): Integer labels array, adjusted for zero-based indexing.
        one_hot_y (torch.Tensor): One-hot encoded labels tensor.
        labels_dict (dict): Dictionary of ImageNet class labels.
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
            return np.array([]), np.array([])  # Early exit if cloning fails

    # Load labels from CSV into a dictionary
    labels_df = pd.read_csv(label_file_path)
    labels_dict = pd.Series(labels_df.TrueLabel.values, index=labels_df.ImageId).to_dict()

    images = []
    labels = []

    # Iterate over files in the images folder
    for filename in os.listdir(images_folder_path):
        if filename.endswith(".png") and filename not in ["362e4ac62cf888f4.png", "bd3617fcc985fe31.png"]:
            img_id = filename[:-4]  # Remove '.png'
            img_path = images_folder_path / filename

            # Read and convert image to RGB
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            images.append(img)

            # Append corresponding label or None if not found
            labels.append(labels_dict.get(img_id, None))

    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    y -= 1   # To make labels correspond to names
    
    # Load ImageNet class labels
    LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    labels_response = requests.get(LABELS_URL)
    labels_dict = labels_response.json()

    y_long = torch.tensor(y, dtype=torch.long)
    one_hot_y = F.one_hot(y_long, num_classes = 1000).float()

    return X, y, one_hot_y, labels_dict