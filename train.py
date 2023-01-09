# importa le librerie necessarie
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import io, datasets, transforms, models
from torchvision.transforms import functional as TF
from torchvision.models import vgg16, VGG16_Weights
from skimage.color import rgb2lab, lab2rgb

# set variabili globali
trainig_session = 
b_size = 
trainset_size = 
validationset_size = 
num_epochs = 
image_size = (224, 224)
model_of_choice = 'ResEnc'
weight_file = 'res_enc.tar'

# Funzioni per la conversione RGB<=>Lab
# Wrapper di rgb2lab
# input:  Tensor {3, h, w} float32 RGB
# output: Tensor {3, h, w} float32 Lab
def RGB2Lab(RGB):
    Lab = RGB.permute(1, 2, 0)
    Lab = np.array(Lab)
    Lab = rgb2lab(Lab).astype("float32")
    Lab = TF.to_tensor(Lab)
    return Lab

# Wrapper di lab2rgb
# input:  Tensor {3, h, w} float32 Lab
# output: Tensor {3, h, w} float32 RGB
def Lab2RGB(Lab):
    RGB = Lab.permute(1, 2, 0)
    RGB = np.array(RGB)
    RGB = lab2rgb(RGB)
    RGB = TF.to_tensor(RGB)
    return RGB

# Scomposizione di Lab nei tensori L (canale L) e ab (canali ab)
# con range dei valori in (-1, 1)
def Lab2L_ab(Lab):
    L = Lab[[0], ...] #/ 50. - 1.
    ab = Lab[[1, 2], ...] #/ 110.
    return L, ab

# Ricomposizione di un'immagine Lab a partire dai canali L e ab
def L_ab2Lab(L, ab):
    Lab = []
    L = L[0]
    a = ab[0]
    b = ab[1]
    Lab.append(L)
    Lab.append(a)
    Lab.append(b)
    Lab = torch.stack(Lab, 0)
    return Lab

# Carica delle immagini da degli indirizzi hardcoded in un Tensor batch
def loadBatch():
    img0 = io.read_image('images/inputs/spiaggia.jpg')
    img1 = io.read_image('images/inputs/faro.jpg')
    img2 = io.read_image('images/inputs/teatro.jpg')
    img3 = io.read_image('images/inputs/foresta.jpg')
    img4 = io.read_image('images/inputs/castello.jpg')
    img5 = io.read_image('images/inputs/facciata.jpg')
    img6 = io.read_image('images/inputs/treno.jpg')
    img7 = io.read_image('images/inputs/orto.jpg')
    img8 = io.read_image('images/inputs/server.jpg')
    img9 = io.read_image('images/inputs/edificio.jpg')
    batch = []
    batch.append(img0)
    batch.append(img1)
    batch.append(img2)
    batch.append(img3)
    batch.append(img4)
    batch.append(img5)
    batch.append(img6)
    batch.append(img7)
    batch.append(img8)
    batch.append(img9)
    for i in range(0, len(batch)):
        batch[i] = TF.resize(batch[i], image_size)
        batch[i] = TF.convert_image_dtype(batch[i])
    batch = torch.stack(batch, 0)
    return batch

# Conversione di una batch di immagini RGB in Lab 
# analogo a RGB2Lab, ma per batch
def batchRGB2Lab(batch):
    batch_size = batch.size()[0]
    for i in range(0, batch_size):
        batch[i] = RGB2Lab(batch[i])
    return batch

# Conversione di una batch di immagini Lab in RGB 
# analogo a Lab2RGB, ma per batch
def batchLab2RGB(batch):
    batch_size = batch.size()[0]
    for i in range(0, batch_size):
        batch[i] = Lab2RGB(batch[i])
    return batch

# analogo a Lab2L_ab ma per batch di immagini Lab
def batchLab2L_ab(bLab):
    bsize = bLab.size()[0]
    bL = []
    bab = []
    for Lab in range(bsize):
        L, ab = Lab2L_ab(bLab[Lab])
        bL.append(L)
        bab.append(ab)
    bL = torch.stack(bL, 0)
    bab = torch.stack(bab, 0)
    return bL, bab

# analogo a L_ab2Lab, ma per batch di canali L e ab
def batchL_ab2Lab(bL, bab):
    batch_size = bL.size()[0]
    batch = []
    for i in range (0, batch_size):
        Lab = L_ab2Lab(bL[i], bab[i])
        batch.append(Lab)
    batch = torch.stack(batch, 0)
    return batch
    
# Salva una batch di Tensor float32 rappresentanti immagiiRGB in memoria come jpeg
def saveBatch(batch, epoch):
    batch_size = batch.size()[0]
    for i in range(0, batch_size):
        img = TF.convert_image_dtype(batch[i], torch.uint8)
        path = '../assets/output' + str(epoch) + '-' + str(i) + '.jpeg'
        io.write_jpeg(img, path)
    return

# hardware detect
# export HSA_OVERRIDE_GFX_VERSION=10.3.0  <---- Fix per la mia GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# preparazione di train_loader e validation_loader
# dichiara trasformazione
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(image_size),
                                transforms.Lambda(RGB2Lab)
                            ])

# dati training
trainset = datasets.SUN397("../data", download=False, transform=transform)
splittrain = torch.utils.data.random_split(trainset, [trainset_size, len(trainset) - trainset_size])[0]
train_loader = DataLoader(splittrain, batch_size=b_size, shuffle=True)

# dati validation
validationset = datasets.SUN397("../data", download=False, transform=transform)
splitvalidation = torch.utils.data.random_split(validationset, [validationset_size, len(trainset) - validationset_size])[0]
validation_loader = DataLoader(splitvalidation, batch_size=b_size, shuffle=True)

# MODELLO DEL RESIDUAL ENCODER ----------------------------------------------------------------------
# step intermedi per vgg 
# ritorna una lista dei risultati dei layer specificati 
# usare una lista permette di eseguire vgg una sola volta
# https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2 
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(weights='DEFAULT').features)[:23]
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
    def __init__(self, input_size = image_size[0] * image_size[1], output_size = image_size[0] * image_size[1] * 3):
        super().__init__()
        # input layer
        self.inconv = nn.Conv2d(1, 3, 1)
        # layer 4
        self.bnorm_4 = nn.BatchNorm2d(512)
        self.conv_4 = nn.Conv2d(512, 256, 1)
        # layer 3
        self.bnorm_3 = nn.BatchNorm2d(256)
        self.conv_3 = nn.Conv2d(256, 128, 3)
        # layer 2
        self.bnorm_2 = nn.BatchNorm2d(128)
        self.conv_2 = nn.Conv2d(128, 64, 3)
        # layer 1
        self.bnorm_1 = nn.BatchNorm2d(64)
        self.conv_1 = nn.Conv2d(64, 3, 3)
        # layer 0
        self.bnorm_0 = nn.BatchNorm2d(3)
        self.conv_0 = nn.Conv2d(3, 3, 3)
        # output
        self.out_conv = nn.Conv2d(3, 2, 1)

    def forward(self, x):
        # istanza di vgg
        vgg = Vgg16().to(device)
        # freeze di vgg
        for child in vgg.children():
            for param in child.parameters():
                param.requires_grad = False
        # input layer
        x = self.inconv(x)
        # forward in vgg-16
        vgg_res = vgg.forward(x)
        vgg_res[0] = x
        # layer 4
        x = vgg_res[4]
        x = self.bnorm_4(x)
        x = self.conv_4(x)
        # layer 3
        x = TF.resize(x, (56, 56))
        x = torch.add(x, self.bnorm_3(vgg_res[3]))
        x = self.conv_3(x)
        # layer 2
        x = TF.resize(x, (112, 112))
        x = torch.add(x, self.bnorm_2(vgg_res[2]))
        x = self.conv_2(x)
        # layer 1
        x = TF.resize(x, (224, 224))
        x = torch.add(x, self.bnorm_1(vgg_res[1]))
        x = self.conv_1(x)
        # layer 0
        x = TF.resize(x, (224, 224))
        x = torch.add(x, self.bnorm_0(vgg_res[0]))
        x = self.conv_0(x)
        # output layer
        x = self.out_conv(x)
        x = TF.resize(x, (224, 224))
        return x
# ----------------------------------------------------------------------

# carica progressi del training
if (model_of_choice == 'ResEnc'): 
    model = ResidualEncoder()
    state_dict = torch.load(weight_file)
    model.load_state_dict(state_dict)
else: model = None
model.to(device)

# salva pesi del modello
def modelSave(model):
    state_dict = model.state_dict()
    if (model_of_choice == 'ResEnc'):
        torch.save(state_dict, weight_file)
    print("Stato del modello salvato")

# preparazione di optimizer e criterion per la loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss().to(device)

# funzione train
def train(epoch, log_interval=100):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Copia data in device
        data = data.to(device)
        # Zero gradient buffers
        optimizer.zero_grad() 

        # ottieni batch di L e ab da data
        L, target_ab = batchLab2L_ab(data)
        L = L.to(device)
        target_ab = target_ab.to(device)

        # predizione immagine a colori
        inferred_ab = model(L).to(device)

        # Calculate loss
        loss = criterion(inferred_ab, target_ab)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

# funzione validate
def validate(loss_vector):
    model.eval()
    val_loss = 0
    for batch_idx, (data, labels) in enumerate(validation_loader):
        data = data.to(device)
        
        L, target_ab = batchLab2L_ab(data)
        L = L.to(device)
        target_ab = target_ab.to(device)
        inferred_ab = model(L).to(device)

        val_loss += criterion(inferred_ab, target_ab).data.item()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    
    print('\nValidation set: Average loss: {:.4f}\n'.format(
        val_loss))

# visualizza i progressi del training sempre sulla stessa immagine per un confronto diretto
def imgValidate(epoch):
    model.eval()
    batch = loadBatch()
    batch = batchRGB2Lab(batch)
    bL, bab = batchLab2L_ab(batch)
    bL = bL.to(device)
    inferred_ab = model(bL).to(device)

    bL = bL.detach().cpu()
    inferred_ab = inferred_ab.detach().cpu()
    batch = batchL_ab2Lab(bL, inferred_ab)
    batch = batchLab2RGB(batch)
    saveBatch(batch, epoch)
    return

# processo di training
print("Dispositivo rilevato: ", device)
epochs = num_epochs
lossv = []
print("Inizio training")
print("Il seguente warning può essere ignorato")
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv)
    imgValidate(epoch)
    modelSave(model)