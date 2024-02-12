import os
import pandas as pd
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels from a specified directory
    and annotations file. This dataset is specifically designed for use with image data, where the
    images are stored as .png files and are loaded into memory as tensors with a datatype of torch.uint8.
    """
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Initializes the dataset by loading annotations and preparing a mapping
        from image IDs to labels.

        Args:
            annotations_file (string): Path to the CSV file with annotations. The CSV file should 
                include 'ImageId' and 'TrueLabel' columns.
            img_dir (string): Directory path containing the image files. Each image file should be named with
                its corresponding 'ImageId' from the annotations_file and should be in .png format.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        # Read the CSV file
        img_labels_df = pd.read_csv(annotations_file, usecols=['ImageId', 'TrueLabel'])
        
        # Create a dictionary mapping from ImageId to TrueLabel
        self.img_labels_dict = pd.Series(img_labels_df.TrueLabel.values, index=img_labels_df.ImageId).to_dict()
        
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels_dict)

    def __getitem__(self, idx):
        # Get the image ID and label from the dictionary
        img_id, label = list(self.img_labels_dict.items())[idx]
        label = int(label) - 1  # Convert to 0-based index if necessary
        
        # Construct the full path to the image file
        img_path = os.path.join(self.img_dir, img_id + '.png')
        image = read_image(img_path)  # Load the image tensor

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        return image, label, img_id
    
def deprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Deprocesses an image that has been processed with the specified normalization.

    Parameters:
        image (torch.Tensor): The processed image tensor to be deprocessed.
        mean (list): The mean used in normalization for each channel.
        std (list): The standard deviation used in normalization for each channel.

    Returns:
        torch.Tensor: The deprocessed image tensor.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(image.device)

    # Inverse of normalization: multiply by std and add the mean
    deprocessed_img = image * std + mean

    # Clip values to ensure they are between 0 and 1
    if not torch.equal(deprocessed_img, torch.clamp(deprocessed_img, 0, 1)):
        print("Deprocessed image has values outside the range [0, 1].")

    return deprocessed_img.to('cpu')