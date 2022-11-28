# importa le librerie necessarie
import torch
import numpy as np
from torchvision import io, transforms

# set variabili
image_size = (224, 224)

# prepara il dataset

# ridimensiona immagine con img = transforms.functional.resize(img, (244, 244))

# converti da RGB a grayscale

# converti da RGB a YUV e viceversa (molto naive, da rivedere)
# https://www.pcmag.com/encyclopedia/term/yuvrgb-conversion-formulas
def rgb2yuv(rgb):
    ch, h, w = rgb.size()
    yuv = rgb.clone()
    
    for i in range(h):
        for j in range(w):
            R = rgb[0][i][j]
            G = rgb[1][i][j]
            B = rgb[2][i][j]

            Y = 0.299*R + 0.587*G + 0.114*B

            yuv[0][i][j] = Y                    # Y
            yuv[1][i][j] = 0.492*(B-Y)          # U
            yuv[2][i][j] = 0.877*(R-Y)          # V

    return yuv

def yuv2rgb(yuv):
    ch, h, w = yuv.size()
    rgb = yuv.clone()

    for i in range(h):
        for j in range(w):
            Y = yuv[0][i][j]
            U = yuv[1][i][j]
            V = yuv[2][i][j]

            R = Y + 1.140*V
            G = Y - 0.395*U - 0.581*V
            B = Y + 2.032*U
    
    return rgb

#

# funzione train

# funzione validate

# training process

# test code here
rgb = io.read_image('images/macchu.jpg')
rgb = transforms.functional.resize(rgb, image_size)
yuv = rgb2yuv(rgb)
con = yuv2rgb(yuv)
io.write_jpeg(rgb, "images/copy.jpeg")
io.write_jpeg(yuv, "images/yuv.jpeg")
io.write_jpeg(rgb, "images/rgb.jpeg")