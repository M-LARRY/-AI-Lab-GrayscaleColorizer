import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from collections import namedtuple
from torchvision.transforms.functional import resize

# variabili conf
img_h = 244
img_w = 244

# step intermedi per vgg 
# ritorna una lista dei risultati dei layer specificati 
# usare una lista permette di eseguire vgg una sola volta
# https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2 
class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(weights=VGG16_Weights.DEFAULT).features)[:23]
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {0,3,8,15,22}:     # specifica qui i layer di cui interessa l'output
                results.append(x)

# modello del Residual Encoder 
# https://tinyclouds.org/colorize
class ResidualEncoder(nn.Module):
    def __init__(self,input_size = img_h * img_w, output_size = img_h * img_w * 3):
        # layer 4 (batchNorm - 1x1Conv)
        self.bnorm_4 = nn.BatchNorm1d(512)
        self.conv_4 = nn.Conv2d(512, 256, 1)
        # layer 3
        self.bnorm_3 = nn.BatchNorm1d(256)
        self.conv_3 = nn.Conv2d(256, 128, 3)
        # layer 2

        # layer 1

        # layer 0

        super().__init__()

    def forward(self, x):
        # forward in vgg-16
        vgg_res = Vgg16().forward(x)
        x = vgg_res[0]
        # layer 4
        # batch norm, 1x1 conv 
        x = self.bnorm_4(x)
        x = self.conv_4(x)
        # layer 3
        # upscale, batch norm, add, 3x3 conv
        x = resize(x, (56, 56))
            # somma di tensori x e self.bnorm_3(vgg_res[1])
        x = self.conv_3(x)
        # layer 2
        # layer 1
        # layer 0
        return

# CODE REVIEW
# 1. Quali layer intermedi scegliere da VGG-16: prima o dopo RELU?
# 2. Sto usando gli argomenti corretti per i layer del modello?
# 3. resize funziona per tensori di profondità maggiore di 3?
# 4. Da riga 4; Credo che questo import non serva 