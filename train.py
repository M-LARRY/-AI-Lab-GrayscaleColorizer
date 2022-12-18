# importa le librerie necessarie
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import io, datasets, transforms, models
from torchvision.transforms import functional as TF

import utils
from residual_encoder import ResidualEncoder

# set variabili
image_size = (224, 224)
trainset_size = 10
validationset_size = 1
model_of_choice = 'ResEnc'  # 'ResEnc' o 'resnet'
loss_func_of_choice = 'MSE'  # 'BUVL' o 'MSE'
num_epochs = 5

# rileva hardware
# export HSA_OVERRIDE_GFX_VERSION=10.3.0  <---- Fix per la mia GPU
def hardware_detect():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

# prepara il dataset
def dataset_setup():
    # dichiara trasformazione
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(image_size),
                                    transforms.Lambda(utils.RGB2YUV),
                                    #transforms.Normalize((0.5,), (0.5,)),   # crea artefatti sull'immagine
                                ])
    # dati training
    trainset = datasets.SUN397("../data", download=False, transform=transform)
    splittrain = torch.utils.data.random_split(trainset, [trainset_size, len(trainset) - trainset_size])[0]
    train_loader = DataLoader(splittrain, batch_size=5, shuffle=True)
    # dati validation
    validationset = datasets.SUN397("../data", download=False, transform=transform)
    splitvalidation = torch.utils.data.random_split(validationset, [validationset_size, len(trainset) - validationset_size])[0]
    validation_loader = DataLoader(splitvalidation, batch_size=5, shuffle=True)
    return train_loader, validation_loader

# carica / salva modello
def load_model(model_of_choice, device):
    if (model_of_choice == 'ResEnc'): 
        model = ResidualEncoder()
        state_dict = torch.load("res_enc.tar")
        model.load_state_dict(state_dict)
    else: model = models.resnet50(weights='DEFAULT', progress=True)
    return model.to(device)

def save_model(model):
    state_dict = model.state_dict()
    if(model_of_choice == 'ResEnc'):
        torch.save(state_dict, "res_enc.tar")

# scegli optimizer e criterion per la loss
def load_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return optimizer

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

def load_criterion(loss_func_of_choice, device):
    if (loss_func_of_choice == 'BUVL'): criterion = None # incompleto
    elif (loss_func_of_choice == 'MSE'): criterion = torch.nn.MSELoss()
    return criterion.to(device)

# funzione train
def train(epoch, log_interval=2):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # ottieni da data inferred_UV e target_UV
        Y, target_UV = utils.batchYUVsplit(data)
        Y = Y.to(device)
        target_UV = target_UV.to(device)

        # infer canali UV
        inferred_UV = model(Y)
        inferred_UV = inferred_UV.to(device)

        # Calculate loss
        loss = criterion(inferred_UV, target_UV)

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
    val_loss, correct = 0, 0
    for data in validation_loader:
        data = data[0].to(device)
        Y, target_UV = utils.batchYUVsplit(data)
        Y = Y.to(device)
        target_UV = target_UV.to(device)
        inferred_UV = model(Y).to(device)
        val_loss += criterion(inferred_UV, target_UV).data.item()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    
    print('\nValidation set: Average loss: {:.4f}\n'.format(
        val_loss))
    
    out = utils.batchYUVjoin(Y, inferred_UV)
    print("punto 1")
    out = out[0]
    print("punto 2")
    out = utils.YUV2RGB(out)
    print("punto 3")
    io.write_jpeg(out, "images/validate.jpeg")

# training process

print("Rilevamento hardware... ", end="")
device = hardware_detect()
print("Fatto")
print("Dispositivo rilevato: ", device)
print('Dataset setup... ', end="")
train_loader, validation_loader = dataset_setup()
print('Fatto')
model = load_model(model_of_choice, device)
optimizer = load_optimizer(model)
criterion = load_criterion(loss_func_of_choice, device)
epochs = num_epochs
lossv = []
print("Inizio training")
print("Il seguente warning può essere ignorato")
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv)
save_model(model)
print("Stato del modello salvato")

# Model forward
'''
print("Rilevamento hardware... ", end="")
device = hardware_detect()
print("Fatto")
print("Dispositivo rilevato: ", device)
model = load_model(model_of_choice, device)
print("Elaboro input...", end="")
img = io.read_image('images/grayscale.jpeg')
img = TF.convert_image_dtype(img, torch.float32)
img = utils.RGB2YUV(img)
img = torch.stack([img], 0)
img, uv = utils.batchYUVsplit(img)
img = img.to(device)
print("Fatto")
inferred_UV = model(img)
print("Preparazione output...", end="")
out = utils.batchYUVjoin(img, inferred_UV)
out = out[0]
out = utils.YUV2RGB(out)
print("Fatto")
io.write_jpeg(out, "images/out.jpeg")
'''
# test code here
'''
img = io.read_image('images/macchu.jpg')
img = TF.resize(img, image_size)
img = TF.convert_image_dtype(img, torch.float32)

yuv = utils.RGB2YUV(img)
y, uv = utils.YUVsplit(yuv)

print("Canale Y è stato separato!")

out = utils.YUVjoin(y, uv)
print("Immagine YUV ricomposta!")

io.write_jpeg(utils.Y2RGB(y), "images/grayscale.jpeg")
io.write_jpeg(utils.YUV2RGB(out), "images/output.jpeg")
'''
'''
print('inizio training...')
epochs = 10

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)

state_dict = model.state_dict()
torch.save(state_dict, "res_enc.tar")

# provo un forward
for batch_idx, (data, labels) in enumerate(train_loader):
    print('Batch',batch_idx,':')
    # ottieni da data inferred_UV e target_UV
    Y, target_UV = utils.batchYUVsplit(data)
    Y = Y.to(device)
    inferred_UV = model(Y)
    # componi un tensore di risultati
    inferred = utils.batchYUVjoin(Y, inferred_UV)

    for img in range(1):   
        io.write_jpeg(TF.convert_image_dtype(utils.Y2RGB(Y[img]), torch.uint8), "images/gscale" + str(img) + ".jpeg")
        io.write_jpeg(TF.convert_image_dtype(inferred[img], torch.uint8), "images/inferred" + str(img) + ".jpeg")
'''
