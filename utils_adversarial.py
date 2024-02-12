import torch



def replace_pixels(xadv, xclean, center, method = "square", cam = None, threshold = None, side_length_square = None, side_length_threshold = None):

    if len(xadv.shape) == 4:
      xadv = xadv.squeeze()

    if len(xclean.shape) == 4:
      xclean = xclean.squeeze()

    if method == "square":

      _, h, w = xadv.shape
      grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h))

      # Calculate distances from the center
      dist_x = torch.abs(grid_x - center[1])
      dist_y = torch.abs(grid_y - center[0])

      # Create mask for the square region around (x, y)
      mask = (dist_x <= side_length_square) & (dist_y <= side_length_square)

      # Expand the mask to have the same number of channels as B
      mask_expanded = mask.unsqueeze(0).expand_as(xadv)

      # Create a new tensor by combining values from A and B based on the mask
      result = torch.where(mask_expanded, xclean, xadv)

    elif method == "threshold":

      # Create coordinates grid
      h, w = cam.shape
      grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h))

      # Calculate distances from the center
      dist_x = torch.abs(grid_x - center[1])
      dist_y = torch.abs(grid_y - center[0])

      # Create mask for the square region around (x, y)
      mask = (dist_x <= side_length_threshold) & (dist_y <= side_length_threshold) & (cam > threshold)

      # Expand the mask to have the same number of channels as A and B
      mask_expanded = mask.unsqueeze(0).expand_as(xadv)

      # Create a new tensor by combining values from A and B based on the mask
      result = torch.where(mask_expanded, xclean, xadv)

    return result.unsqueeze(0)


def MFGSM(x_clean, y_true, model_name, model, eps, T, alpha,mean, std, momentum = None, device = "cuda"):

  y_true = y_true.to(device)

  if momentum != None:
    use_momentum = True
  else:
    use_momentum = False

  x_adv = torch.clone(x_clean).detach().requires_grad_(True)

  criterion = torch.nn.CrossEntropyLoss()

  normalized_clamp = Normalized_Clamp(mean, std, device)

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

def ADP(x_clean, y_true, model_name, model, ratio_threshold, min_distance, eps, T, alpha, beta, m, mean, std, cam_method = "normal", momentum = None, device = "cuda", use_zero_gradient_method = False, replace_method = "square"):

  y_true = y_true.to(device)

  if momentum != None:
    use_momentum = True
  else:
    use_momentum = False

  target_layer = get_target_layer(model_name, model)

  if cam_method == "normal":
    cam = GradCAM(model=model, target_layers=target_layer)

  elif cam_method == "++":
    cam = GradCAMPlusPlus(model=model, target_layers=target_layer)

  targets = [ClassifierOutputTarget(torch.argmax(y_true))]

  x_adv = torch.clone(x_clean).detach().requires_grad_(True)

  W, H = x_clean.shape[1], x_clean.shape[2]

  criterion = torch.nn.CrossEntropyLoss()

  normalized_clamp = Normalized_Clamp(mean, std, device)

  for t in range(0, T):

    x_adv.requires_grad_(True)

    g = 0

    if use_momentum:
      previous_g = 0

    M = cam(input_tensor=x_adv, targets=targets)

    max_cam = M.max()

    centers = get_centers(M[0], ratio_threshold, min_distance)

    for center in centers:

      for k in range(1, m+1):
        #x1 = int(max(center[0]-beta*k, 0))
        #x2 = int(min(center[0]+beta*k, W))
        #y1 = int(max(center[1]-beta*k, 0))
        #y2 = int(min(center[1]+beta*k, H))

        if not use_zero_gradient_method:

          x_drop = torch.clone(x_adv).detach()

          if replace_method == "square":
            x_drop.data = replace_pixels(x_drop.data, x_clean.data, center, "square", side_length_square = beta*k)

          elif replace_method == "threshold":
            x_drop.data = replace_pixels(x_drop.data, x_clean.data, center, "threshold", M[0], threshold = max_cam/(0.88**k), side_length_threshold = beta*m)

          #x_drop.data[:, x1:x2, y1:y2] = x_clean.data[:, x1:x2, y1:y2]
          x_drop = x_drop.detach()
          x_drop.requires_grad_(True)

          loss = criterion(model(x_drop)[0], y_true)
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
            grad.data = replace_pixels(grad.data, torch.zeros_like(grad.data), center, "threshold", M[0], threshold = max_cam/(0.88**k), side_length_threshold = beta*m)

          #grad[:, x1:x2, y1:y2] = 0
          g += grad

          x_adv.grad.zero_()

    g *= 1/(len(centers)*m)

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