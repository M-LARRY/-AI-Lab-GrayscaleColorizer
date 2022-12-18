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
    YUV = torch.zeros(3, h, w, dtype=torch.float32)
    
    for i in range(h):
        for j in range(w):
            R = RGB[0][i][j]
            G = RGB[1][i][j]
            B = RGB[2][i][j]

            Y = (0.299*R) + (0.587*G) + (0.114*B)

            YUV[0][i][j] = Y                    # Y
            YUV[1][i][j] = 0.492*(B-Y)          # U
            YUV[2][i][j] = 0.877*(R-Y)          # V

    return YUV

# ritorna una copia dell'immagine YUV convertita in RGB ATTENZIONE PROBABILE ERRORE DI SEGMENTAZIONE QUI!!!
def YUV2RGB(YUV):
    # nota: R,G,B hanno range [0, 255]
    ch, h, w = YUV.size()
    RGB = torch.zeros(3, h, w, dtype=torch.float32)
    
    for i in range(h):
        for j in range(w):
            Y = YUV[0][i][j] + 1
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

# ritorna dei nuovi canali UV data un'immagine RGB
def RGB2UV(RGB):
    YUV = RGB2YUV(RGB)
    Y, UV = YUVsplit(YUV)
    return UV

# ritorna una copia di YUV in scala di grigi in formato RGB (funzione dalla dubbia utitlità)
def yuv2gscale(yuv):
    ch, h, w = yuv.size()
    gscale = torch.zeros(1, h, w, dtype=torch.float32)

    for i in range(h):
        for j in range(w):
            gscale[0][i][j] = yuv[0][i][j]
    
    return TF.convert_image_dtype(gscale, torch.uint8)

# ritorna una copia di Y in formato RGB (nel modo meno efficiente possibile, lazy)
def Y2RGB(Y):
    ch, h, w = Y.size()
    return YUV2RGB(YUVjoin(Y, torch.zeros(2, h, w, dtype=torch.float32)))

# ritorna un tensore di canali Y e un tensore di canali UV a partire da una batch di immagini YUV
def batchYUVsplit(batch_YUV):
    size, ch, h, w = batch_YUV.size()
    Y_list = []
    UV_list = []
    for batch_idx in range(0, size):
        Y, UV = YUVsplit(batch_YUV[batch_idx])
        Y_list.append(Y)
        UV_list.append(UV)
    batch_Y = torch.stack(Y_list, 0)
    batch_UV = torch.stack(UV_list, 0)
    return batch_Y, batch_UV

# ritorna un tensore batch contenente l'unione dei canali Y con i rispettivi UV
def batchYUVjoin(batch_Y, batch_UV):
    size, ch, h, w = batch_Y.size()
    YUV_list = []
    for batch_idx in range(0, size):
        YUV = YUVjoin(batch_Y[batch_idx], batch_UV[batch_idx])
        YUV_list.append(YUV)
    batch_YUV = torch.stack(YUV_list, 0)
    return batch_YUV