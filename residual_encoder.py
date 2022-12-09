import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from collections import namedtuple
from torchvision.transforms.functional import resize

# variabili conf
img_h = 244
img_w = 244

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# step intermedi per vgg 
# ritorna una lista dei risultati dei layer specificati 
# usare una lista permette di eseguire vgg una sola volta
# https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2 
class Vgg16(nn.Module):
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
        return results

# modello del Residual Encoder 
# https://tinyclouds.org/colorize
class ResidualEncoder(nn.Module):
    def __init__(self, input_size = img_h * img_w, output_size = img_h * img_w * 3):
        super().__init__()
        # input
        self.in_conv = nn.Conv2d(1, 3, 1)
        # layer 4 (batchNorm - 1x1Conv)
        self.bnorm_4 = nn.BatchNorm2d(512)
        self.conv_4 = nn.Conv2d(512, 256, 1)
        # layer 3
        self.bnorm_3 = nn.BatchNorm1d(256)
        self.conv_3 = nn.Conv2d(256, 128, 3)
        # layer 2
        self.bnorm_2 = nn.BatchNorm1d(128)
        self.conv_2 = nn.Conv2d(128, 64, 3)
        # layer 1
        self.bnorm_1 = nn.BatchNorm1d(64)
        self.conv_1 = nn.Conv2d(64, 3, 3)
        # layer 0
        self.bnorm_0 = nn.BatchNorm1d(3)
        self.conv_0 = nn.Conv2d(3, 3, 3)
        # output
        self.out_conv = nn.Conv2d(3, 2, 3)

    def forward(self, x):
        # aumenta i canali di gray input
        x = self.in_conv(x)
        # forward in vgg-16
        vgg = Vgg16().to(device)
        vgg_res = vgg.forward(x)
        # layer 4
        x = vgg_res[4]
        x = self.bnorm_4(x)
        x = self.conv_4(x)
        # layer 3
        x = resize(x, (56, 56))
        torch.add(x, self.bnorm_3(vgg_res[3]), out = x)     # attualmente l'esecuzione termina qui
        x = self.conv_3(x)
        # layer 2
        x = resize(x, (112, 112))
        torch.add(x, self.bnorm_2(vgg_res[2]), out = x)
        x = self.conv_2(x)
        # layer 1
        x = resize(x, (224, 224))
        torch.add(x, self.bnorm_1(vgg_res[1]), out = x)
        x = self.conv_1(x)
        # layer 0
        torch.add(x, self.bnorm_0(vgg_res[0]), out = x)
        x = self.conv_0(x)
        # output layer
        x = self.out_conv(x)
        return x

# CODE REVIEW
# 1. Quali layer intermedi scegliere da VGG-16: prima o dopo RELU?
# 2. Sto usando gli argomenti corretti per i layer del modello?
# 3. resize funziona per tensori di profondità maggiore di 3? (dalla documentazione sembra di sì)
# 4. Da riga 4; Credo che questo import non serva 