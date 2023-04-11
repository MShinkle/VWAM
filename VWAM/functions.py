import torch
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from collections import OrderedDict
from torch.utils.data import Dataset
import glob
from PIL import Image

def choose_downsampling(activations, max_fs):
    """Selects max pooling downsampling for model layers based on multiple heuristics.

    Args:
        activations (torch.Tensor): Layer activations from a single forward pass.
        max_fs (int): Maximum number of desired downsampled features per layer.

    Returns:
        None or torch.nn.MaxPool2D: Downsampling function selected for the layer.  If activations.ndim != 4 (as in fc layers), will return None.
            Otherwise, will select 2D max pooling such that:
                Total # of downsampled features <= 10000
                Stride is <= kernel size.
            If multiple combinations of stride and kernel size fulfill these requirements, the combo which get closes to 10000 is chosen.
    """    
    if activations.ndim == 4:
        test_range = activations.shape[-1]
        numels = np.zeros((test_range+1, test_range))
        for k in range(1,test_range+1):
            for s in range(1,k+1):
                n = (activations.shape[-1] - k) / s
                if n != int(n):
                    continue
                else:
                    pooled = torch.nn.functional.max_pool2d(activations, kernel_size=k, stride=s)
                    if pooled.shape[-1] > 1 and pooled.numel() <= max_fs:
                        numels[k,s] = pooled.numel()
                    else:
                        continue
        best_k, best_s = np.unravel_index(np.argmax(numels, axis=None), numels.shape)
        if (best_k, best_s) == (0,0):
            return None
        else:
            return torch.nn.MaxPool2d(kernel_size=(best_k, best_k), stride=best_s)
    else:
        return None

def show_imgs(imgs, titles=None, show=True, axs=None):
    """Displays pytorch tensors as images.

    Args:
        imgs (torch.Tensor): PyTorch tensor of images of shape (# images, 3, height, width)
        titles (list of str, optional): List of string titles, one for each image/subplot. Defaults to None.
        show (bool, optional): Whether to call plt.show(). Defaults to True.
        axs (matplotlib.axes, optional): Matplotlib axes on which to show images. Defaults to None.
    """    
    if isinstance(axs, type(None)):
        fig, axs = plt.subplots(nrows=1, ncols=len(imgs), figsize=(5*len(imgs), 5))
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs, dtype='object')
    imgs = np.clip(imgs.detach().cpu().numpy().astype(np.float),0,1)
    for i, ax in zip(range(len(imgs)), axs.flatten()):
        ax.imshow(np.moveaxis(imgs[i], 0, 2), vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[i], color='darkslategrey')
    if show:
        plt.show()

def get_rf_mask(model, layer_name, indices, input_shape=(1,3,224,224), device='cpu'):
    """Uses gradients to estimate the pixel receptive field for unit(s) in a model.

    Args:
        model (hooked pytorch model): PyTorch model passed through hook_model, containing layer and unit(s) for which to estimate RF.
        layer_name (str): Name of layer for which to compute RF; should be a key in model.activations after forward pass.
        indices (int or tuple of ints): Unit indices for which to compute combined RF.  E.g. 5 will compute the RF for fifth unit of layer layer_name,
        and (3,5,10) will compute the RF for units 3, 5 and 10 combined.
        input_shape (tuple, optional): Size of input to model. Defaults to (1,3,224,224).
        device (str, optional): Device on which to run model. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Mask of shape input_shape, with ones for pixels which fall within the unit RF, and zeros elsewhere.
    """    
    model = model.to(device)
    if not isinstance(indices, tuple):
        indices = tuple(indices)
    init_image = torch.zeros(input_shape, dtype=torch.float).to(device).requires_grad_(True)
    optimizer = torch.optim.SGD([init_image], lr=1e10,)
    optimizer.zero_grad()
    model(init_image);
    loss = torch.moveaxis(torch.moveaxis(model.activations[layer_name], 0, -1)[indices], -1, 0)
    loss.backward()
    mask = init_image.grad.detach() != 0
    return mask

def compute_diversity(activations):
    """Compute a differentiable diversity metric for the difference between two or more images,
    computed as the mean pairwise cosine similarity between layer responses to each image.

    Args:
        activations (torch.Tensor): Tensor of size (n_images, n_units) containing reponses of units or layers
            to two or more images.

    Returns:
        float: 'Diversity' metric across images.  Will be in range(-1,1), with -1 meaning identical activations for all images
            (no diversity) and 1 meaning completely orthogonal vectors.
    """    
    dot_prod = activations@activations.T
    activations_norm = torch.linalg.norm(activations, dim=1)
    norm_outer = torch.outer(activations_norm, activations_norm)
    cos_similarity = dot_prod / norm_outer
    return -torch.nanmean(cos_similarity[~torch.eye(len(cos_similarity), dtype=bool)]).item()

def pink_noise(shape, power=1, fft=False):
    """Produce a tensor of specified shape containing noise with a specified distribution.

    Args:
        shape (tuple of ints): Desired shape of output noise tensor.
        power (int, optional): Power of the distance from the center (denominator term).  E.g. 1 will yield approximately
            pink noise, 0 white noise and 2 Brownian noise. Defaults to 1.
        fft (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: tensor of specified shape composed of noise following the specified distribution.
    """    
    white_noise = torch.randn(shape)
    x, y = torch.meshgrid(torch.linspace(-10,10,shape[-2]), torch.linspace(-10,10,shape[-1]))
    dist = torch.sqrt(x**2 + y**2)
    white_fft = torch.fft.fft2(white_noise)
    pink_fft = white_fft * 1/dist**power
    if fft:
        return pink_fft
    else:
        pink_noise = torch.abs(torch.fft.ifft2(pink_fft))
        return pink_noise

def generate_one_over_n(size, power=1, device='cpu'):
    """Produce a 2D distribution displaying a specified relationship between distance from the center and magnitude.
        Designed for use with spect_dist, in which a computed frequency disribution is compared to an idealized one. 

    Args:
        size (int): Dimension of desired distribution, which will be (size, size)
        power (int, optional: Power of the distance from the center (denominator term); higher values will result in quicker dropoff. 
            Defaults to 1, which produces a 1/f distribution.
        device (str, optional): Device onto which returned tensor should be placed. Defaults to 'cpu'.

    Returns:
        torch.Tensor: size X size tensor containing specified distribution.  Normalized to sum to 1.
    """    
    dists = np.sqrt(np.broadcast_to(np.linspace(-1,1,size), (size,size))**2 + np.broadcast_to(np.linspace(-1,1,size), (size,size)).T**2)
    ideal_power_2D = 1/dists**power
    ideal_power_2D /= ideal_power_2D.sum()
    return torch.tensor(ideal_power_2D, device=device)

def spect_dist(image, ideal_spect):
    """A way to compare two different 2D frequency distributions.  Computed as the euclidean norm of the difference
        between the normalized (sum to 1) fourier distribution and ideal_spect.

    Args:
        image (torch.Tensor): 2D image of same shape as ideal spect.
        ideal_spect (torch.Tensor): Desired frequency distribution, such as the output of generate_one_over_n

    Returns:
        float: Euclidean norm of the difference between the normalized (sum to 1) fourier distribution and ideal_spect.
    """    
    power_2D = torch.fft.fft2(image)
    power_2D = torch.fft.fftshift(power_2D, dim=(-1,-2))
    power_2D = torch.abs(power_2D)
    power_2D /= power_2D.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    return torch.linalg.norm((ideal_spect - power_2D).flatten(), ord=2)

# Values acquired from lucent/lucid, originally computed from ImageNet
color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
color_mean = [0.48, 0.46, 0.41]

def linear_decorrelate_color(image):
    """Modifies input image based on Cholesky-based color decorrelation used in Olah, et al. (2017).

    Args:
        image (torch.tensor): Image(s) on which to perform decorrelation.  Should be of shape (# images, 3, height, width)

    Returns:
        torch.Tensor: Decorrelated version of input image.
    """
    t_permute = image.permute(0, 2, 3, 1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T, dtype=t_permute.dtype).to(t_permute.device))
    t_permute += torch.tensor(color_mean, dtype=t_permute.dtype).to(t_permute.device)
    image = t_permute.permute(0, 3, 1, 2)
    return image

def compute_argmax_layer(weights, trn_means):
    """Determine which layer is assigned highest summed regression weights.

    Args:
        weights (torch.Tensor): Tensor of regression weights for every unit across all layers.
            Should be of length equal to the summed lengths of all values of trn_means.
        trn_means (dict of pytorch tensors): Dictionary in form of {layer name: layer activations}.
            Total number of elements across all values should be equal to the length of weights.

    Returns:
        int: Index of layer assigned the highest summed regression weights.
    """    
    layer_weights_list = []
    i = 0
    for layer_name, layer_activations in trn_means.items():
        layer_size = np.prod(layer_activations.shape)
        layer_weights_list.append(weights[i:i+layer_size].sum(0))
        i += layer_size
    argmax_layer = np.nanargmax(layer_weights_list, 0)
    return argmax_layer
