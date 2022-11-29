# importa le librerie necessarie
import torch
import numpy as np
from torchvision import io
from torchvision.transforms import functional as TF
import utils

# set variabili
image_size = (224, 224)

# prepara il dataset

# ridimensiona immagine con img = transforms.functional.resize(img, (244, 244))
# estrai i canali Y e UV ( usa YUVsplit() )

#

# funzione train

# funzione validate

# training process

# test code here
img = io.read_image('images/macchu.jpg')
img = TF.resize(img, image_size)

yuv = utils.RGB2YUV(img)
y, uv = utils.YUVsplit(yuv)

print("Canale Y è stato separato!")

out = utils.YUVjoin(y, uv)
print("Immagine YUV ricomposta!")

io.write_jpeg(utils.Y2RGB(y), "images/grayscale.jpeg")
io.write_jpeg(utils.YUV2RGB(out), "images/output.jpeg")