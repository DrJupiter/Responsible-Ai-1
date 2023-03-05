from model import Net_Logistic
import torch
from dataload import CatalanDataset, datasplit, convert_dataload 
import numpy as np
import random

# Reducability
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
EPOCHS = 700

# Initialize data
dataset = CatalanDataset('./data/preprocessed.csv')
train, validation, test = convert_dataload(datasplit(dataset))
_x, _y = next(iter(train))

# Initialize model
model = Net_Logistic(_x.size()[1]) 
model= torch.nn.DataParallel(model)
model.to(DEVICE)
model.train()

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
loss_fn = torch.nn.BCELoss()

# Train
losses = []
accur = []

for i in range(EPOCHS):
  for j,(feature,target) in enumerate(train):
    feature, target = feature.to(DEVICE), target.to(DEVICE)
    #calculate output
    output = model(feature)
 
    #calculate loss
    loss = loss_fn(output,target.reshape(-1,1))
 
    #accuracy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
val_loss = []
for feature, target in validation:
    feature, target = feature.to(DEVICE), target.to(DEVICE)

    val_loss.append((model(feature).round() == target).reshape(-1).detach().cpu().numpy())

print(np.mean(val_loss))