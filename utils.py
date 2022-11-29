import torch
import numpy as np
from torchvision import io
from torchvision.transforms import functional as TF

# converti da RGB a YUV e viceversa (vorrei usare un solo tensore per yuv e tmp, ma causo un bug. Purtroppo così è particolarmente lenta)
# ritorna una copia dell'immagine RGB convertita in YUV
def RGB2YUV(RGB):
    # nota: Y ha range [0,1]           Rappresenta la luminanza
    #       U ha range [-0.436, 0.436]
    #       V ha range [-0.615, 0.615]
    ch, h, w = RGB.size()
    tmp = TF.convert_image_dtype(RGB, torch.float32)
    YUV = torch.zeros(3, h, w, dtype=torch.float32)
    
    for i in range(h):
        for j in range(w):
            R = tmp[0][i][j]
            G = tmp[1][i][j]
            B = tmp[2][i][j]

            Y = (0.299*R) + (0.587*G) + (0.114*B)

            YUV[0][i][j] = Y                    # Y
            YUV[1][i][j] = 0.492*(B-Y)          # U
            YUV[2][i][j] = 0.877*(R-Y)          # V

    return YUV

# ritorna una copia dell'immagine YUV convertita in RGB
def YUV2RGB(YUV):
    # nota: R,G,B hanno range [0, 255]
    ch, h, w = YUV.size()
    RGB = torch.zeros(3, h, w, dtype=torch.float32)
    
    for i in range(h):
        for j in range(w):
            Y = YUV[0][i][j] + 1    # perchè funziona? non lo so...
            U = YUV[1][i][j]
            V = YUV[2][i][j]

            RGB[0][i][j] = Y + (1.140*V)                 # R 
            RGB[1][i][j] = Y - (0.395*U) - (0.581*V)     # G
            RGB[2][i][j] = Y + (2.033*U)                 # B
    
    return TF.convert_image_dtype(RGB, torch.uint8)

# ritorna una copia del canale Y ed una copia dei canali UV
def YUVsplit(YUV):
    ch, h, w = YUV.size()
    Y = torch.zeros(1, h, w, dtype=torch.float32)
    UV = torch.zeros(2, h, w, dtype=torch.float32)

    for i in range(w):
        for j in range(h):
            Y[0][i][j] = YUV[0][i][j]
            UV[0][i][j] = YUV[1][i][j]
            UV[1][i][j] = YUV[2][i][j]

    return Y, UV

# ritorna una nuova immagine YUV a partire da Y e UV
def YUVjoin(Y, UV):
    ch, h, w = Y.size()
    YUV = torch.zeros(3, h, w, dtype=torch.float32)
    
    for i in range(w):
        for j in range(h):
            YUV[0][i][j] = Y[0][i][j]
            YUV[1][i][j] = UV[0][i][j]
            YUV[2][i][j] = UV[1][i][j]

    return YUV

# ritorna una copia di YUV in scala di grigi in formato RGB (funzione dalla dubbia utitlità)
def yuv2gscale(yuv):
    ch, h, w = yuv.size()
    gscale = torch.zeros(1, h, w, dtype=torch.float32)

    for i in range(h):
        for j in range(w):
            gscale[0][i][j] = yuv[0][i][j]
    
    return TF.convert_image_dtype(gscale, torch.uint8)

# ritorna una copia di Y in formato RGB
def Y2RGB(Y):
    ch, h, w = Y.size()
    return YUV2RGB(YUVjoin(Y, torch.zeros(2, h, w, dtype=torch.float32)))