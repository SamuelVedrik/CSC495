import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from constants import DEVICE
from tqdm import tqdm
from collections import defaultdict

class VGGWrapper(nn.Module):
    def __init__(self, base: nn.Module, num_out: int):
        super().__init__()
        self.vgg = base
        self.vgg.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_out)
    )
        
    def forward(self, x):
        return self.vgg(x)
    
    def freeze_layers(self):
        for param in self.vgg.features.parameters():
                param.requires_grad = False
        
    
    def latent_vars(self, x):
        # Get the features fed into the linear classifier.
        latent = self.vgg.features(x)
        return self.vgg.avgpool(latent)

class VGGBlocksWrapper(VGGWrapper):
    
    ## https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg11
    ## Part of cfs["A"], where each numeric denotes a conv and ReLU layer, and a "M" denotes maxpool.
    BLOCKS = [0, 0, 0, 
              1, 1, 1,
              2, 2, 2, 2, 2,
              3, 3, 3, 3, 3,
              4, 4, 4, 4, 4]
    def __init__(self, base, num_out: int):
        super().__init__(base, num_out)
        self.vgg_layers = defaultdict(list)

        for layer, block_id in zip(self.vgg.features.children(), self.BLOCKS):
            self.vgg_layers[block_id].append(layer)
    
    def get_from_block(self, x: torch.Tensor, block: int):
        modules = []
        for block_id in range(block+1):
            modules.extend(self.vgg_layers[block_id])
        
        layer = nn.Sequential(*modules)    
        result = layer(x)
        result = self.vgg.avgpool(result)
        return result


def get_latent_variables(model, datasets, num_per_class=100, seed=495):
    latent_variables = {}
    for name, dataset in datasets.items():
        small = dataset.split_dataset(num_per_class=num_per_class, seed=seed)
        dataloader = DataLoader(small, batch_size=16, shuffle=False)
        reps_all = []
        labels_all = []
        print(name)
        for images, labels in tqdm(dataloader):
            images = images.to(DEVICE)
            rep = model.latent_vars(images).detach().cpu()
            reps_all.append(rep)
            labels_all.append(labels)
            torch.cuda.empty_cache()
    
        reps_all = torch.cat(reps_all, axis=0)
        labels_all = torch.cat(labels_all, axis=0)
        latent_variables[name] = (reps_all, labels_all)
    return latent_variables


# ======== VARIATIONAL AUTO ENCODER ============ 
class LinearBlock(nn.Module): 
    def __init__(self, in_features, out_features, activation, final_loss=True): 
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            activation(),
            nn.BatchNorm1d(out_features),
        )
        
    def forward(self, x): 
        return self.block(x)
    
class VAE(nn.Module):
    def __init__(self, in_features, dim_red, activation=nn.ReLU): 
        super().__init__()
        self.dim_red = dim_red
        encoder_final = dim_red*2
        self.encoder = nn.Sequential(
            LinearBlock(in_features, 200, activation),
            LinearBlock(200, 200, activation),
            LinearBlock(200, 200, activation),
            LinearBlock(200, 200, activation),
            LinearBlock(200, 200, activation),
            nn.Linear(200, encoder_final)
        )
        
        self.decoder =  nn.Sequential(
            LinearBlock(dim_red, 200, activation),
            LinearBlock(200, 200, activation),
            LinearBlock(200, 200, activation),
            LinearBlock(200, 200, activation),
            LinearBlock(200, 200, activation),
            nn.Linear(200, in_features)
        )
        
    def reparam(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x): 
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.dim_red], encoded[:, self.dim_red:]
        sample = self.reparam(mu, log_var)
        decoded = self.decoder(sample)
        return decoded, mu, log_var

        
    def transform(self, x):
        self.eval()
        with torch.no_grad():
            if type(x) == np.ndarray: 
                x = torch.FloatTensor(x).to(DEVICE)

            return self.encoder(x)[:, :self.dim_red].cpu().numpy()
        

