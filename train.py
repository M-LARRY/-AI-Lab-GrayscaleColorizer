# importa le librerie necessarie
import torch
import numpy as np
from torchvision import io, datasets, transforms, models
from torchvision.transforms import functional as TF

import utils
from residual_encoder import ResidualEncoder

# rileva hardware
# export HSA_OVERRIDE_GFX_VERSION=10.3.0  <---- Fix per la mia GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Versione di PyTorch:', torch.__version__, ' Device:', device)

# set variabili
image_size = (224, 224)
model_of_choice = 'resnet'  # 'ResEnc' o 'resnet'

# prepara il dataset

# dichiara trasformazione
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.Resize(image_size),
                                transforms.Lambda(utils.RGB2YUV),
                              ])

# dati training

trainset = datasets.SUN397("dataset", download=False, transform=transform)
train_loader = DataLoader(trainset, batch_size=10, shuffle=True)
trainset


# dati validation
'''
validationset = datasets.SUN397("dataset", download=False, transform=transform)
validation_loader = DataLoader(validationset, batch_size=10, shuffle=True)
validationset
'''

# dati test
'''
testset = datasets.SUN397("dataset", download=False, transform=transform)
test_loader = DataLoader(testset, batch_size=10, shuffle=True)
testset
'''

# iteratore iter(train_loader)


# estrai i canali Y e UV ( usa YUVsplit() )

# scelta del modello
if (model_of_choice == 'ResEnc'): model = None #ResidualEncoder()
else: model = models.resnet50(weights='DEFAULT', progress=True)

# scegli optimizer e criterion per la loss

# funzione train

# funzione validate

# training process

# test code here

'''
img = io.read_image('images/macchu.jpg')
img = TF.resize(img, image_size)

yuv = utils.RGB2YUV(img)
y, uv = utils.YUVsplit(yuv)

print("Canale Y è stato separato!")

out = utils.YUVjoin(y, uv)
print("Immagine YUV ricomposta!")

io.write_jpeg(utils.Y2RGB(y), "images/grayscale.jpeg")
io.write_jpeg(utils.YUV2RGB(out), "images/output.jpeg")
'''