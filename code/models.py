import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None
    
def grad_reverse(x: torch.Tensor, lambd: float = 1.0):
    return GradientReverseFunction.apply(x, lambd)

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.model= nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size= kernel_size, stride= stride, padding= padding, bias= False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor):
        return self.model(x)

class FCBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout= 0.0):
        super().__init__()
        layers= [nn.Linear(in_dim, out_dim, bias= False), nn.BatchNorm1d(out_dim), nn.ReLU(inplace= True),]
        if dropout> 0:
            layers.append(nn.Dropout(dropout))
        self.model= nn.Sequential(*layers)
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
class Encoder(nn.Module):
    def __init__(self, input_len: int = 300, latent_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
                                                    ConvBlock1D(3, 16, kernel_size=15, stride=1, padding=7), ConvBlock1D(16, 32, kernel_size=15, stride=1, padding=7), nn.MaxPool1d(kernel_size=2),
                                                    ConvBlock1D(32, 64, kernel_size=7, stride=1, padding=3), ConvBlock1D(64, 128, kernel_size=7, stride=1, padding=3), nn.MaxPool1d(kernel_size=2),
                                                    ConvBlock1D(128, 256, kernel_size=3, stride=1, padding=1),
                                                )
        with torch.no_grad():
            dummy= torch.zeros(1, 3, input_len)
            flat_dim= int(np.prod(self.conv(dummy).shape[1: ]))
        self.fc= FCBlock(flat_dim, latent_dim, dropout= dropout)
    def forward(self, x: torch.Tensor):
        z= self.conv(x)
        z= torch.flatten(z, start_dim= 1)
        return self.fc(z)
    
class DANN(nn.Module):
    def __init__(self, crop_len: int, num_classes: int, num_domains: int, latent_dim= 512, fc1= 256, fc2= 100, dropout= 0.0):
        super().__init__()
        self.encoder= nn.ModuleList([Encoder(input_len=crop_len, latent_dim=latent_dim, dropout=dropout) for _ in range(3)])
        fused_dim= latent_dim * 3
        self.predictor= nn.Sequential(FCBlock(fused_dim, fc1, dropout= dropout), FCBlock(fc1, fc2, dropout= dropout), nn.Linear(fc2, num_classes))
        ### adversarial neural network
        self.domain_discriminator = nn.Sequential(FCBlock(fused_dim, fc1, dropout=dropout), FCBlock(fc1, fc2, dropout=dropout), nn.Linear(fc2, num_domains))
    def extract_features(self, x: torch.Tensor):
        features= []
        for crop_idx, extractor in enumerate(self.encoder):
            crop= x[:, crop_idx, :, :]
            features.append(extractor(crop))
        return torch.cat(features, dim=1)
    def forward(self, x: torch.Tensor, grl_lambda= 1.0):
        features= self.extract_features(x)
        class_probability= self.predictor(features)
        domain_probability= self.domain_discriminator(grad_reverse(features, grl_lambda))
        return class_probability, domain_probability, features