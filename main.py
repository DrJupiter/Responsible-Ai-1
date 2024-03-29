#%%
from model import Net_Logistic
import torch
from dataload import CatalanDataset, datasplit, convert_dataload, dataloader_to_dataframe, get_encoding_table
from fairnessmetrics import test_fairness
import numpy as np
import pandas as pd
import random
import time 

import pdb
#%%

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
      self.n_epochs = 1
      self.print_freq = 100


def main():
  CFG = cfg()  

  ### Load data, split and initialize dataloaders ###
  pd_df = pd.read_csv(CFG.data_path)
  dataset = CatalanDataset(pd_df, person_sensitive=True)
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
    print("epoch:",i)
         
    train_loss, train_acc = train(model, DataloaderTrain, optimizer, criterion, CFG)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    val_loss, val_acc, _ = validate(model, DataloaderVal, criterion, CFG)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

  test_losses, test_acc  = [], []
  test_loss, test_acc, predictions = validate(model, DataloaderTest, criterion, CFG)
  test_losses.append(test_loss)
  test_acc.append(test_acc)
  
  # save fairness data in results/fairnes.json
  fairness(model, DataloaderTest, CFG)

  return train_losses, val_losses, train_accs, val_accs, test_losses, test_acc
  
     

def train(model, dataloader, optimizer, criterion, CFG):
  model.train()
  acc = 0
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
    
    acc += accuracy(out.round().reshape(-1).detach().cpu(), trg)
    #pdb.set_trace()
    end = time.time()
    if i % CFG.print_freq == 0:
       print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses):.4f}\nAcc: {acc / (trg.size(0)*i+1):.4f}")
  
  return losses, acc

def validate(model, dataloader, criterion, CFG):
  model.eval()
  acc = 0
  losses = []
  predictions = []
  start = time.time()

  with torch.no_grad():
    for i, (ipt, trg) in enumerate(dataloader):
      ipt = ipt.to(CFG.device)
      trg = trg.to(CFG.device)

      out = model(ipt)
      loss = criterion(out, trg)
      losses.append(loss.cpu())
      predictions.append(out)

      acc += accuracy(out.round().reshape(-1).detach().cpu(), trg)
      end = time.time()
      if i % CFG.print_freq == 0:
        print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses):.4f}\nAcc: {acc/(trg.size(0)*i+1):.4f}")

  return losses, acc, predictions

def fairness(model, dataloader, CFG):
  model.eval()
  predictions = []
  
  with torch.no_grad():
    for i, (ipt, trg) in enumerate(dataloader):
      ipt = ipt.to(CFG.device)
      trg = trg.to(CFG.device)
      
      #np.where(model(ipt).detach().cpu().numpy() <= 0.5, 1, 0)
      predictions.append((model(ipt).round()).reshape(-1).detach().cpu().numpy())
   
  df = dataloader_to_dataframe(dataloader, dataloader.dataset.dataset.columns)

  test_fairness(df, predictions)

def accuracy(pred, trg):
  acc = 0
  for i in range(pred.size(0)):
    if pred[i]==trg[i]:
      acc+=1
  
  return acc

if __name__ == "__main__":
  main()

# %%
