from model import Net_Logistic
import torch
from dataload import CatalanDataset, datasplit, convert_dataload 
import numpy as np
import random
import time 

# Reducability
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class cfg:
   def __init__(self):
      self.data_path = './data/preprocessed.csv'
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.lr = 1e-4
      self.start_epoch = 0
      self.n_epochs = 700
      self.print_freq = 50


def main():
  CFG = cfg()  

  ### Load data, split and initialize dataloaders ###
  dataset = CatalanDataset(CFG.data_path)
  DataloaderTrain, DataloaderVal, DataloaderTest = convert_dataload(datasplit(dataset))
  _x, _y = next(iter(DataloaderTrain))

  ### Initialize model, optimizer and loss ###
  model = Net_Logistic(_x.size()[1]) 
  model= torch.nn.DataParallel(model)
  model.to(CFG.device)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

  criterion = torch.nn.BCELoss().to(CFG.device)

  train_losses, val_losses, train_accs, val_accs = [], [], [], []

  ### Start training loop ###
  for i in range(CFG.start_epoch, CFG.n_epochs):
     
     train_loss, train_acc = train(model, DataloaderTrain, optimizer, criterion, CFG)
     train_losses.append(train_loss)
     train_accs.append(train_acc)

     val_loss, val_acc = validate(model, DataloaderVal, criterion, CFG)
     val_losses.append(val_loss)
     val_accs.append(val_acc)
     

def train(model, dataloader, optimizer, criterion, CFG):
  model.train()
  accs = []
  losses = []
  start = time.time()

  for i, (ipt, trg) in enumerate(dataloader):
    ipt = ipt.to(CFG.device)
    trg = trg.to(CFG.device)

    out = model(ipt)
    loss = criterion(out, trg)
    losses.append(loss.detach().cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end = time.time()
    if i % CFG.print_freq == 0:
       print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses)}")
  return losses, accs

def validate(model, dataloader, criterion, CFG):
  model.eval()
  accs = []
  losses = []
  start = time.time()

  with torch.no_grad():
    for i, (ipt, trg) in enumerate(dataloader):
      ipt = ipt.to(CFG.device)
      trg = trg.to(CFG.device)

      out = model(ipt)
      loss = criterion(out, trg)
      losses.append(loss.cpu())

      end = time.time()
      if i % CFG.print_freq == 0:
        print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses)}")

  return losses, accs



val_loss = []
for feature, target in validation:
    feature, target = feature.to(DEVICE), target.to(DEVICE)

    val_loss.append((model(feature).round() == target).reshape(-1).detach().cpu().numpy())

print(np.mean(val_loss))