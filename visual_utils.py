import matplotlib.pyplot as plt
from constants import INV_NORMALIZE
from typing import Dict
import torch
import plotly.graph_objects as go

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
    
    
def plot_encoded_values(representations, labels, dataset, title=""):
    fig = go.Figure()
    for id, label in dataset.id_to_labels.items():
        points = representations[labels == id]
        fig.add_trace(go.Scatter(
                    x = points[:, 0],
                    y = points[:, 1],
                    mode="markers",
                    name = label
            ))
    fig.update_layout(
        xaxis_range = [representations[:, 0].min() - abs(representations[:, 0].min() * 0.05), representations[:, 0].max() + abs(representations[:, 0].max() * 0.05)],
        yaxis_range = [representations[:, 1].min() - abs(representations[:, 1].min() * 0.05), representations[:, 1].max() + abs(representations[:, 1].max() * 0.05)],
        title = title,
        width=800,
        height=800
    )
    fig.show()