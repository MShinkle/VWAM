import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn, Tensor
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython import display
from functools import partial
import dill as pickle
from collections import defaultdict

def iterate_children(model, mode='children', depth=1):
    modules = [('filler name', model)]
    if mode == 'children':
        for d in range(depth):
            modules = [named_submodule for name, module in modules for named_submodule in module.named_children()]
    elif mode == 'modules':
        modules = model.named_modules()
    for module in modules:
        yield module

def store_activations(activations_dict, layer_name, module, input, output):
    activations_dict[layer_name] = output

def hook_model(model, mode, depth):
    model.activations = defaultdict(list)
    for layer_name, child in iterate_children(model, mode=mode, depth=depth):
        child.register_forward_hook(partial(store_activations, model.activations, layer_name))
    return model

def zscore_layer(trn_mean, trn_std, activations):
    activations = (activations - trn_mean) / trn_std
    return activations

def multiply_weights(weights, activations):
    return weights.unsqueeze(0)*activations.reshape(len(activations), -1)

def show_imgs(imgs, titles=None, show=True, axs=None):
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

def compute_diversity(t):
    dot_prod = t@t.T
    t_norm = torch.linalg.norm(t, dim=0)
    norm_outer = t_norm.T@t_norm
    cos_similarity = dot_prod / norm_outer
    return -torch.nanmean(torch.triu(cos_similarity, diagonal=1))

def generate_one_over_f(size, power=1, device='cpu'):
    dists = np.sqrt(np.broadcast_to(np.linspace(-1,1,size), (size,size))**2 + np.broadcast_to(np.linspace(-1,1,size), (size,size)).T**2)
    ideal_power_2d = 1/dists**power
    ideal_power_2d /= ideal_power_2d.sum()
    return torch.tensor(ideal_power_2d, device=device)

def spect_dist(image, ideal_spect):
    power_2d = torch.fft.fft2(image)
    power_2d = torch.fft.fftshift(power_2d, dim=(-1,-2))
    power_2d = torch.abs(power_2d)
    power_2d /= power_2d.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    return torch.linalg.norm((ideal_spect - power_2d).flatten(), ord=2)

def get_rf_mask(model, layer_name, indices, input_shape=(1,3,224,224), device='cpu'):
    model = model.to(device)
    if not isinstance(indices, tuple):
        indices = tuple(indices)
    init_image = torch.zeros(input_shape, dtype=torch.float).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([init_image], lr=1e10,)
    optimizer.zero_grad()
    model(init_image);
    loss = torch.moveaxis(torch.moveaxis(model.activations[layer_name], 0, -1)[indices], -1, 0)
    loss.backward()
    mask = init_image.grad.detach() != 0
    return mask

def choose_downsampling(activations, max_fs):
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


# Taken directly from lucent
color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]

def _linear_decorrelate_color(tensor):
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T, dtype=t_permute.dtype).to(t_permute.device))
    t_permute += torch.tensor(color_mean, dtype=t_permute.dtype).to(t_permute.device)
    tensor = t_permute.permute(0, 3, 1, 2)
    return tensor

class  LinearDecorrelateColor(object):
    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = _linear_decorrelate_color(sample,
        #  device=self.device
        )
        return image

def compute_argmax_layer(weights, trn_means):
    layer_weights_list = []
    i = 0
    for layer_name, layer_activations in trn_means.items():
        layer_size = np.prod(layer_activations.shape)
        layer_weights_list.append(weights[i:i+layer_size].sum(0))
        i += layer_size
    argmax_layer = np.nanargmax(layer_weights_list, 0)
    return argmax_layer

def pink_noise(shape, power=1, fft=False):
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