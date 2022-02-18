from torchvision import transforms
import torch

RESIZE = transforms.Resize((224, 224))

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
