import glob
from PIL import Image
from functools import partial
from collections import defaultdict
from torch.utils.data import Dataset

class SingleImageFolder(Dataset):
    def __init__(self, parent_dir, transform=None):
        self.image_paths = sorted(glob.glob(parent_dir+'/*'))
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

def iterate_children(child, parent_name='model', depth=1, keep_dead_ends=True):
    named_grandchildren = list(child.named_children())
    if len(named_grandchildren) == 0 and keep_dead_ends:
        return {parent_name: child}
    else:
        if depth > 1:
            children_dict = dict()
            for name, grandchild in named_grandchildren:
                children_dict.update(iterate_children(grandchild, parent_name+'.'+name, depth-1))
            return children_dict
        else:
            return {parent_name+'.'+name: module for name, module in named_grandchildren}

def iterate_modules(model):
    return {name: module for name, module in model.named_modules()}

def store_activations(activations_dict, layer_name, module, input, output):
    """Stores activations of an individual layer in a dictionary . Intended to be used with forward hooks.

    Args:
        activations_dict (dict): Dictionary of form {layer name: layer activations}.
        layer_name (str): Name of layer.
        module (_type_): Unused, but required input from forward hook
        input (_type_): Unused, but required input from forward hook
        output (torch.Tensor): Layer activations, to be saved to activations_dict.
    """   
    activations_dict[layer_name] = output

def hook_model(model, layers_dict):
    """Uses forward hooks to modify model to save activations on forward pass.
    Activations are stored in model.activations.

    Args:
        model (pytorch model): NN model onto which hooks will be added.

    Returns:
        pytorch model: Modified version of model which stores activations in model.activations after forward pass.
    """    
    model.activations = defaultdict(list)
    for layer_name, child in layers_dict.items():
        child.register_forward_hook(partial(store_activations, model.activations, layer_name))
    return model