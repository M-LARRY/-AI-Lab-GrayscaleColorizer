# importa le librerie necessarie
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
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
model_of_choice = 'ResEnc'  # 'ResEnc' o 'resnet'
loss_func_of_choice = 'MSE'  # 'BUVL' o 'MSE'

# prepara il dataset
# dichiara trasformazione
transform = transforms.Compose([transforms.ToTensor(),
                                #transforms.Normalize((0.5,), (0.5,)),   # crea artefatti sull'immagine
                                transforms.Resize(image_size),
                                transforms.Lambda(utils.RGB2YUV),
                              ])

print('Carico dataset...')
# dati training
trainset = datasets.SUN397("../data", download=False, transform=transform)
train_loader = DataLoader(trainset, batch_size=1, shuffle=True)
#print(trainset)

# dati validation
validationset = datasets.SUN397("../data", download=False, transform=transform)
validation_loader = DataLoader(validationset, batch_size=10, shuffle=True)

# dati test
testset = datasets.SUN397("../data", download=False, transform=transform)
test_loader = DataLoader(testset, batch_size=10, shuffle=True)

print('dataset caricato!')

# scelta del modello
if (model_of_choice == 'ResEnc'): model = ResidualEncoder()
else: model = models.resnet50(weights='DEFAULT', progress=True)
model.to(device)

# scegli optimizer e criterion per la loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# blur_uv_loss di Dahl, modificata per le mie funzioni
def blur_uv_loss(rgb, inferred_rgb):
  uv = utils.RGB2UV(rgb)
  uv_blur0 = utils.RGB2UV(TF.gaussian_blur(rgb, 3))
  uv_blur1 = utils.RGB2UV(TF.gaussian_blur(rgb, 5))

  inferred_uv = utils.RGB2UV(inferred_rgb)
  inferred_uv_blur0 = utils.RGB2UV(TF.gaussian_blur(inferred_rgb, 3))
  inferred_uv_blur1 = utils.RGB2UV(TF.gaussian_blur(inferred_rgb, 5))

  return  ( torch.cdist(inferred_uv, uv) +
            torch.cdist(inferred_uv_blur0 , uv_blur0) +
            torch.cdist(inferred_uv_blur1, uv_blur1) ) / 3

if (loss_func_of_choice == 'BUVL'): criterion = None # incompleto
elif (loss_func_of_choice == 'MSE'): criterion = torch.nn.MSELoss().to(device)

# funzione train
def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        print(data, labels)
        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # prepara gscale e target
        Y, UV = utils.YUVsplit(data)
        Y = Y.to(device)
        UV = UV.to(device)

        # inferenza di UV
        inferredUV = model(Y)

        # ricomponi immagine
        inferred = utils.YUVjoin(Y, inferredUV)

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
def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data in validation_loader:
        data = data[0].to(device)
        Y, UV = utils.YUVsplit(data)
        Y = Y.to(device)
        UV = UV.to(device)
        inferredUV = model(Y)
        inferred = utils.YUVjoin(Y, inferredUV)
        val_loss += criterion(inferred, data).data.item()
        pred = inferred.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

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

print('inizio training...')
epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)