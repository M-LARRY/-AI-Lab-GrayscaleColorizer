import torch
import torch.nn as nn
from torchvision import models

# variabili conf
img_h = 244
img_w = 244

class ResidualEncoder(nn.Module):
    def __init__(self,input_size = img_h * img_w, output_size = img_h * img_w * 3):
        
        super().__init__()

    def forward(self, x):
        
        return


residualEncoder = ResidualEncoder()
print(residualEncoder)