import torchvision.models
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from utils_dataset import CustomImageDataset

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

class CombinedModel(nn.Module):
    def __init__(self, models):
        super(CombinedModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        total_output = torch.sum(torch.stack(outputs), dim=0)
        return total_output


def get_model(model_name):
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
    if model_name == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.input_size = 224
        model.target_layers = [model.layer4[-1]]  # Directly assign target layers
    elif model_name == "squeezenet":
        model = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT)
        model.input_size = 224
        model.target_layers = [model.features[-1]]
    elif model_name == "resnet101":
        model = timm.create_model("resnet101", pretrained=True)
        model.input_size = 224
        model.target_layers = [get_module_by_name(model, 'layer4')[-1]]
    elif model_name == "inception_v3":
        model = timm.create_model("inception_v3", pretrained=True)
        model.input_size = 299
        model.target_layers = [get_module_by_name(model, 'Mixed_7c')]
    elif model_name == "inception_v4":
        model = timm.create_model("inception_v4", pretrained=True)
        model.input_size = 299
        model.target_layers = [get_module_by_name(model, 'features')[-1]]
    elif model_name == "adv_inception_v3":
        model = timm.create_model("inception_v3.tf_adv_in1k", pretrained=True)
        model.input_size = 299
        model.target_layers = [get_module_by_name(model, 'Mixed_7c')]
    elif model_name == "inception_resnet_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
        model.input_size = 299
        model.target_layers = [get_module_by_name(model, 'conv2d_7b')]
    # elif model_name == "ensemble":
    #     inception_v3 = get_model("inception_v3")
    #     inception_v4 = get_model("inception_v4")
    #     inception_resnet_v2 = get_model("inception_resnet_v2")
    #     model = CombinedModel([inception_v3, inception_v4, inception_resnet_v2])
    else:
        raise ValueError(f"Unsupported model name '{model_name}'. Please provide a valid model name.")

    # Define and attach the transform attribute for preprocessing
    model.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(model.input_size, model.input_size)),
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model.name = model_name
    model.eval()
    model.requires_grad_(False)  # Freeze parameters
    
    return model

def load_model(model_name, model_dir="Model"):
    """
    Checks if a model is saved in the specified directory and loads it. If not,
    it downloads the model (when online), saves it, and then loads it.

    Parameters:
        model_name (str): The name of the model to load or save.
        model_dir (str, optional): The directory to check for the saved model and
                                   where to save the model. Defaults to "Model".

    Returns:
        model (torch.nn.Module): The loaded PyTorch model.
    """
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    
    # Check if the model already exists
    if os.path.isfile(model_path):
        print(f"Loading model '{model_name}' from {model_path}")
        model = torch.load(model_path)
    else:
        print(f"Model '{model_name}' not found in {model_dir}. Downloading and saving...")
        model = get_model(model_name)  # Replace this with actual download function
        torch.save(model, model_path)
        print(f"Model '{model_name}' saved to {model_path}")
    
    return model

def evaluate_model_accuracy(model, dataloader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = model.transform(images).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of {model.name}: {accuracy}%')

    model.to('cpu')
    torch.cuda.empty_cache()

    return accuracy


def evaluate_cross_model_accuracy(models, label_file_path, batch_size, num_workers, device, dir_suffix="_apd"):
    """
    Evaluates the accuracy of models against adversarial examples generated for each other and compiles the results into a DataFrame.

    Parameters:
        models (list): A list of models to evaluate.
        label_file_path (str): Path to the file containing labels.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for the DataLoader.
        device (str): The device to run the evaluation on.
        dir_suffix (str): Suffix for the directory containing adversarial images.

    Returns:
        pd.DataFrame: A pivot table showing the accuracy of each base model against each target model.
    """
    results = []

    for model_base in (m for m in models if m.name != "adv_inception_v3"):
        images_folder_path = os.path.join("Generated", f"{model_base.name}{dir_suffix}")
        dataset = CustomImageDataset(label_file_path, images_folder_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        for model_target in models:
            print(f"Attacking {model_target.name} with {model_base.name}")
            accuracy = evaluate_model_accuracy(model_target, dataloader, device)
            
            results.append({
                "Base Model": model_base.name,
                "Target Model": model_target.name,
                "Accuracy": accuracy
            })

    df_results = pd.DataFrame(results)
    df_pivot = df_results.pivot("Base Model", "Target Model", "Accuracy")

    print(df_pivot)
    return df_pivot