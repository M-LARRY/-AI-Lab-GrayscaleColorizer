# importa le librerie necessarie
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import io, datasets, transforms, models
from torchvision.transforms import functional as TF
from torchvision.models import vgg16, VGG16_Weights

# set variabili globali
trainig_number = 1
b_size = 10
trainset_size = 1000
validationset_size = 100
num_epochs = 15
image_size = (224, 224)
model_of_choice = 'ResEnc'  # 'ResEnc' o 'unet'

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
        features = list(vgg16(VGG16_Weights.DEFAULT).features)[:23] # trova il modo di freezare questi pesi in training (dovrebbe essere zero_grad o requires_grad)
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
    def __init__(self, input_size = image_size[0] * image_size[1] * 3, output_size = image_size[0] * image_size[1] * 3):
        super().__init__()
        # layer 4 (batchNorm - 1x1Conv)
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
        self.out_conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        # forward in vgg-16
        vgg = Vgg16().to(device).requires_grad_(False)
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
    state_dict = torch.load("res_enc.tar")
    model.load_state_dict(state_dict)
else: model = None
model.to(device)

# preparazione di optimizer e criterion per la loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss().to(device)

# funzione train
def train(epoch, log_interval=5):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        # Zero gradient buffers
        optimizer.zero_grad() 

        # ottieni grayscale da data
        grayscale = TF.rgb_to_grayscale(data, 3).to(device)

        # predizione immagine a colori
        inferred = model(grayscale).to(device)

        # Calculate loss
        loss = criterion(inferred, data)

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
        
        grayscale = TF.rgb_to_grayscale(data, 3).to(device)
        inferred = model(grayscale).to(device)

        val_loss += criterion(inferred, data).data.item()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    
    print('\nValidation set: Average loss: {:.4f}\n'.format(
        val_loss))

# visualizza i progressi del training sempre sulla stessa immagine per un confronto diretto
def imgValidate(epoch):
    img = io.read_image('images/macchu.jpg')
    img = TF.resize(img, image_size)
    img = TF.convert_image_dtype(img, torch.float32)
    img = TF.rgb_to_grayscale(img, 3).to(device)
    img = torch.stack([img], 0)
    img = model(img).cpu()
    img = img[0]
    img = TF.convert_image_dtype(img, torch.uint8).cpu()
    path = "images/validation/output" + str(epoch * trainset_size * trainig_number) + ".jpeg"
    io.write_jpeg(img, path)

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

# salvataggio dei progressi
state_dict = model.state_dict()
if(model_of_choice == 'ResEnc'):
    torch.save(state_dict, "res_enc.tar")
    print("Stato del modello salvato")