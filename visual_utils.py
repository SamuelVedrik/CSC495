import matplotlib.pyplot as plt
from constants import INV_NORMALIZE
from typing import Dict
import torch

plt.rcParams["figure.figsize"] = (13, 7)

def show_image(image, normalized=True, ax=None, title=""):
    """
    Plots an image from a tensor.
    """
    if normalized: 
        image = torch.clip(INV_NORMALIZE(image), 0, 1)
    image = image.detach().cpu().permute(1, 2, 0)
    if ax:
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)
    else: 
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)
        
def showcase_transforms(datasets: Dict, idx=0):
    fig, axs = plt.subplots(nrows=3, ncols=4)
    dataset_names = list(datasets.keys())
    for i in range(4):
        for j in range(3):
            name = dataset_names[(i*3)+j]
            show_image(datasets[name][idx][0], ax=axs[j][i], title=name)
    fig.tight_layout()
    plt.show()
    