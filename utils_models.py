import torchvision.models
import timm
import torch

def get_module_by_name(model, module_name):
    """
    Retrieves a specific module from a PyTorch model by its name.

    Parameters:
        model (torch.nn.Module): The model to search through.
        module_name (str): The name of the module to retrieve.

    Returns:
        torch.nn.Module or None: The requested module if found; otherwise, None.
    """
    for name, module in model.named_modules():
        if name.endswith(module_name): 
            return module
    return None

def get_model(model_name, device):
    """
    Loads a specified model and prepares it for inference, with parameters frozen.
    Additionally, attaches 'transform' and 'target_layers' attributes to the model for required preprocessing
    and target layer(s) identification.

    Parameters:
        model_name (str): The name of the model to load.
        device (torch.device or str): The device to transfer the model to ('cpu', 'cuda').

    Returns:
        torch.nn.Module: The requested model, ready for inference, with 'transform' and 'target_layers' attributes.
    """
    # Instantiate the model
    if model_name == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.input_size = 224
        model.target_layers = [model.layer4[-1]]  # Directly assign target layers
    elif model_name == "squeezenet":
        model = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)
        model.input_size = 224
        model.target_layers = [model.features[-1]]
    elif model_name == "inception_v4":
        model = timm.create_model("inception_v4", pretrained=True)
        model.input_size = 299
        model.target_layers = [get_module_by_name(model, 'features')[-1]]
    elif model_name == "adv_inception_v3":
        model = timm.create_model("inception_v3.tf_adv_in1k", pretrained=True)
        model.input_size = 299
        model.target_layers = [get_module_by_name(model, 'Mixed_7c')]
    else:
        raise ValueError(f"Unsupported model name '{model_name}'. Please provide a valid model name.")

    # Define and attach the transform attribute
    model.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(model.input_size, model.input_size)),
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model.name = model_name
    model.to(device).eval()
    model.requires_grad_(False)  # Freeze parameters

    return model
