import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils_cam import get_centers
import os
from tqdm import tqdm
from torchvision.utils import save_image


def MFGSM(x_clean, y_true, model_name, model, eps, T, alpha,
          mean, std, momentum = None, device = "cuda"):

  y_true = y_true.to(device)

  if momentum != None:
    use_momentum = True
  else:
    use_momentum = False

  x_adv = torch.clone(x_clean).detach().requires_grad_(True)

  criterion = torch.nn.CrossEntropyLoss()

  if use_momentum:
    previous_g = 0

  g = 0

  for t in range(0, T):

    x_adv.requires_grad_(True)

    loss = criterion(model(x_adv)[0], y_true)
    loss.backward()
    g = torch.clone(x_adv.grad.data)
    x_adv.grad.zero_()

    if use_momentum:
      g = momentum * previous_g + g / torch.mean(torch.abs(g), dim = (1,2,3), keepdim=True)

    x_adv_max = x_clean + eps
    x_adv_min = x_clean - eps

    with torch.no_grad():

      x_adv_max = normalized_clamp(x_adv_max)
      x_adv_min = normalized_clamp(x_adv_min)

      g_sign = g.sign()
      perturbed_x_adv = x_adv + alpha * g_sign
      x_adv = torch.max(torch.min(perturbed_x_adv, x_adv_max), x_adv_min)

      previous_g = torch.clone(g)

  return x_adv

def normalized_clamp(normalized_tensor,
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Clamps a batch of normalized tensors or a single normalized tensor to ensure 
    that, when unnormalized, its values are within the valid pixel range [0, 1].
    
    Parameters:
        normalized_tensor (torch.Tensor): The tensor or batch of tensors to clamp,
            assumed to be normalized with the provided mean and std. The tensor 
            shape can be either (C, H, W) for a single image or (N, C, H, W) for 
            a batch of images, where N is the batch size.
        mean (list): The mean used in normalization for each channel.
        std (list): The standard deviation used in normalization for each channel.
    Returns:
        torch.Tensor: The clamped tensor or batch of tensors, with values adjusted 
            to ensure that its unnormalized form will have pixel values within [0, 1].

    Note:
        The function assumes the input tensor(s) is already normalized. The clamping 
        is applied based on the reverse normalization equation to ensure valid 
        unnormalized pixel values.
    """
    mean = torch.tensor(mean, device=normalized_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=normalized_tensor.device).view(1, 3, 1, 1)

    low = (0 - mean) / std
    high = (1 - mean) / std

    clamped_tensor = torch.clamp(normalized_tensor, low, high)
    return clamped_tensor


def replace_pixels(xadv, xclean, center, method="square", cam=None, threshold=None, 
                   side_length_square=None, side_length_threshold=None):
    """
    Replaces pixels in an adversarial example with pixels from the clean image
    based on a specified method and conditions.
    """
    # Ensure xadv and xclean are 4D (batch, channel, height, width)
    if xadv.dim() != 4 or xclean.dim() != 4:
        raise ValueError("xadv and xclean must have dimensions [B, C, H, W]")

    B, C, H, W = xadv.shape
    device = xadv.device
    
    # Generate grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    if method == "square":
        # Calculate distances from the center
        dist_x = torch.abs(grid_x - center[1])
        dist_y = torch.abs(grid_y - center[0])
        
        # Create mask for the square region around the center
        mask = (dist_x <= side_length_square) & (dist_y <= side_length_square)
        
    elif method == "threshold":
        if cam is None or threshold is None or side_length_threshold is None:
            raise ValueError("For 'threshold' method, cam, threshold, and side_length_threshold must be provided.")
        
        # Ensure cam matches the spatial dimensions
        if cam.shape[-2:] != (H, W):
            raise ValueError("Spatial dimensions of cam must match xadv and xclean.")
        
        # Calculate distances and create mask based on threshold
        dist_x = torch.abs(grid_x - center[1])
        dist_y = torch.abs(grid_y - center[0])
        mask = (dist_x <= side_length_threshold) & (dist_y <= side_length_threshold) & (cam > threshold)
    
    else:
        raise ValueError("Unsupported method specified.")
    
    # Expand mask to cover channels and batch
    mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    
    # Apply mask to combine xclean and xadv
    result = torch.where(mask_expanded, xclean, xadv)

    return result

def APD(x_clean, y_true, model, 
        min_distance=20, eps=0.274, T=10, alpha=0.5, beta=15, m=5,
        ratio_threshold=0.6, cam_method = "++", momentum = None,
        use_zero_gradient_method = False, replace_method = "square"):
  """
  Performs Adversarial Perturbation Dropout (APD) to generate adversarial examples.

  Parameters:
      x_clean (torch.Tensor): The clean input image tensor.
      y_true (torch.Tensor): The true label tensor for x_clean.
      model (torch.nn.Module): The model against which the adversarial attack is performed.
      min_distance (int): Minimum distance between centers for dropout.
      eps (float): The maximum perturbation allowed.
      T (int): Number of iterations to perform the attack.
      alpha (float): Step size for each iteration.
      beta (int): Parameter to control the size of the square/region for replacement.
      m (int): Number of dropout iterations per main iteration.
      ratio_threshold (float): Ratio threshold for selecting center points based on CAM.
      cam_method (str): Method for CAM generation, supports "normal" and "++".
      momentum (float or None): Momentum factor for gradient accumulation. If None, momentum is not used.
      use_zero_gradient_method (bool): If True, uses zero gradient method for dropout.
      replace_method (string): Method to replace pixels, supports "square" and "threshold".

  Returns:
      torch.Tensor: The adversarial example generated from the clean image.
  """
  device = next(model.parameters()).device
  target_layers = model.target_layers
  if cam_method == "normal":
    cam = GradCAM(model=model, target_layers=target_layers)
  elif cam_method == "++":
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
  targets = [ClassifierOutputTarget(torch.argmax(y_true))]

  y_true = y_true.to(device)
  x_clean_preprocessed = model.transform(x_clean).to(device)
  x_adv = x_clean_preprocessed.clone().detach().requires_grad_(True)
  
  criterion = torch.nn.CrossEntropyLoss().to(device)
  
  for t in range(0, T):
    x_adv.requires_grad_(True)
    g = 0
    if momentum is not None:
      previous_g = 0
    grayscale_cam = cam(input_tensor=x_adv, targets=targets)
    max_cam = grayscale_cam.max()
    centers = get_centers(grayscale_cam[0], ratio_threshold, min_distance)

    for center in centers:
      for k in range(1, m+1):
        if not use_zero_gradient_method:
          x_drop = torch.clone(x_adv).detach()
          if replace_method == "square":
            x_drop.data = replace_pixels(x_drop.data, x_clean_preprocessed.data, center, 
                                         "square", side_length_square = beta*k)
          elif replace_method == "threshold":
            x_drop.data = replace_pixels(x_drop.data, x_clean_preprocessed.data, center, 
                                         "threshold", grayscale_cam[0], 
                                         threshold = max_cam/(0.88**k), 
                                         side_length_threshold = beta*m)
          x_drop = x_drop.detach()
          x_drop.requires_grad_(True)
          loss = criterion(model(x_drop), y_true)
          loss.backward()
          grad = x_drop.grad.data
          g += grad
          x_drop.grad.zero_()

        elif use_zero_gradient_method:
          loss = criterion(model(x_adv)[0], y_true)
          loss.backward()
          grad = x_adv.grad.data
          if replace_method == "square":
            grad.data = replace_pixels(grad.data, torch.zeros_like(grad.data), center, "square", side_length_square = beta*k)
          elif replace_method == "threshold":
            grad.data = replace_pixels(grad.data, torch.zeros_like(grad.data), center, "threshold", grayscale_cam[0],
                                       threshold = max_cam/(0.88**k), side_length_threshold = beta*m)
          g += grad
          x_adv.grad.zero_()

    g *= 1/(len(centers)*m)

    if momentum is not None:
      g = momentum * previous_g + g / torch.mean(torch.abs(g), dim = (1,2,3), keepdim=True)

    x_adv_max = x_clean_preprocessed + eps
    x_adv_min = x_clean_preprocessed - eps

    with torch.no_grad():
      x_adv_max = normalized_clamp(x_adv_max)
      x_adv_min = normalized_clamp(x_adv_min)
      g_sign = g.sign()
      perturbed_x_adv = x_adv + alpha * g_sign
      x_adv = torch.max(torch.min(perturbed_x_adv, x_adv_max), x_adv_min)
      previous_g = torch.clone(g)

  return x_adv


def save_adversarial_images(dataloader, model, device, save_dir=None):
    """
    Processes the entire dataset to generate and save adversarial images within a 'Generated'
    parent directory, naming the subdirectory after the model if not specified.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset, expected to return
                                                  batches in the format of (images, labels, IDs).
        model (torch.nn.Module): The model used to generate adversarial examples.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') for computation.
        save_dir (str, optional): Subdirectory within 'Generated' where adversarial images will be saved.
                                  If None, a directory named '{model.name}_adv' is used.

    Returns:
        None. Saves adversarial images to disk, skipping if the directory already exists.
    """
    if save_dir is None:
        save_dir = f"{model.name}_adv"
    save_dir = os.path.join("Generated", save_dir)  # Prepend 'Generated' to the save_dir

    if os.path.exists(save_dir):
        print(f"Directory '{save_dir}' already exists. Skipping adversarial image generation and saving.")
        return

    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    msg = f"Generating adversarial images with {model.name}"
    for batch_idx, (images, labels, IDs) in enumerate(tqdm(dataloader, desc=msg)):
        # Ensure images and labels are on the same device as the model
        x_adv = APD(images.to(device), labels.to(device), model, device=device)  # Adjust as necessary
        x_adv_cpu = x_adv.cpu()
        for i, x_adv_img in enumerate(x_adv_cpu):
            save_path = os.path.join(save_dir, f"{IDs[i]}.png")
            save_image(x_adv_img, save_path)

    model.to('cpu')
    print(f"Adversarial images saved in '{save_dir}'.")
