from model import Net_Logistic
from feature_model import FeatureModel
import torch
import torch.nn as nn
from dataload import CatalanDataset, datasplit, convert_dataload 
import numpy as np
import pandas as pd
import random
import time 


class cfg:
   def __init__(self):
      self.data_path = './data/preprocessed.csv'
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.lr = 1e-4
      self.start_epoch = 0
      self.n_epochs = 700
      self.print_freq = 50

def jointLoss(joint_preds, g1_preds, g2_preds, trg):
  group_criterion = torch.log.nn.BCELoss()



def main():
  CFG = cfg()  

  ### Load data, split and initialize dataloaders ###
  dataset = CatalanDataset(pd.read_csv(CFG.data_path))
  DataloaderTrain, DataloaderTrainG1, DataloaderTrainG2, DataloaderVal, DataloaderTest = convert_dataload(datasplit(dataset))
  _x, _y = next(iter(DataloaderTrain))

  ### Initialize model, optimizer and loss ###
  joint_model = Net_Logistic(_x.size()[1])
  feature_model = FeatureModel(_x.size()[1])
  model_g1 = Net_Logistic(_x.size()[1])
  model_g2 = Net_Logistic(_x.size()[1])

  joint_model.to(CFG.device)
  feature_model.to(CFG.device)
  model_g1.to(CFG.device)
  model_g2.to(CFG.device)

  joint_optimizer = torch.optim.Adam(joint_model.parameters(), lr=CFG.lr)
  feature_optimizer = torch.optim.Adam(feature_model.parameters(), lr=CFG.lr)
  optimizer_g1 = torch.optim.Adam(model_g1.parameters(), lr=CFG.lr)
  optimizer_g2 = torch.optim.Adam(model_g2.parameters(), lr=CFG.lr)

  joint_train_losses, joint_val_losses, joint_train_accs, joint_val_accs = [], [], [], []
  feature_train_losses, feature_val_losses, feature_train_accs, feature_val_accs = [], [], [], []
  train_losses_g1, val_losses_g1, train_accs_g1, val_accs_g1 = [], [], [], []
  train_losses_g2, val_losses_g2, train_accs_g2, val_accs_g2 = [], [], [], []

  ### Define custom loss functions ###
  criterion_m = nn.BCELoss(reduction='mean')
  criterion_s = nn.BCELoss(reduction='sum')

  def Ld(out,target):
    #criterion_s
    l = []
    for out_i, target_i in zip(out,target):
      li = torch.log(criterion_m(out_i,target_i))
      l.append(li)
    L = torch.tensor(l).mean()
    return -L

  def L0(out,target):
    L = criterion_m(out,target) 
    return L

  def LR(out: tuple(torch.Tensor,torch.Tensor),target):
    out = out[0]
    out_t = out[1]
    l = []
    for out_i, out_t_i, target_i in zip(out,out_t,target):
      li = torch.log(criterion_m(out_i,target_i))-torch.log(criterion_m(out_t_i,target_i))
      l.append(li)
    L = torch.tensor(l).mean()
    return 0.7*L + L0(out,target)

  ### Start training loop ###
  for i in range(CFG.start_epoch, CFG.n_epochs):
     
    # train and validate for group 1
    train_loss_g1, train_acc_g1 = train(model_g1, DataloaderTrainG1, optimizer_g1, Ld, CFG)
    train_losses_g1.append(train_loss_g1)
    train_accs_g1.append(train_acc_g1)

    val_loss_g1, val_acc_g1 = validate(model_g1, DataloaderVal, Ld, CFG)
    val_losses_g1.append(val_loss_g1)
    val_accs_g1.append(val_acc_g1)

    # train and validate for group 2
    train_loss_g2, train_acc_g2 = train(model_g2, feature_model, DataloaderTrainG2, optimizer_g2, Ld, CFG)
    train_losses_g2.append(train_loss_g2)
    train_accs_g2.append(train_acc_g2)
    val_loss_g2, val_acc_g2 = validate(model_g2, feature_model, DataloaderVal, Ld, CFG)
    val_losses_g2.append(val_loss_g2)
    val_accs_g2.append(val_acc_g2)
    
    # feature model training
    train_loss_feature, train_acc_feature = train(feature_model, None, DataloaderTrain, feature_optimizer, Ld, CFG)
    val_loss_feature, val_acc_feature = validate()
    

    # joint model training
    train_loss, train_acc = train(joint_model, feature_model, DataloaderTrain, joint_optimizer, LR, CFG)
    joint_train_losses.append(train_loss)
    joint_train_accs.append(train_acc)

    val_loss, val_acc = validate(joint_model, feature_model, DataloaderVal, L0, CFG)
    joint_val_losses.append(val_loss)
    joint_val_accs.append(val_acc)


def train(model, feature_model, dataloader, optimizer, criterion, CFG, train_type='feature'):
  model.train()
  accs = []
  losses = []
  start = time.time()

  for i, (ipt, trg) in enumerate(dataloader):
    ipt = ipt.to(CFG.device)
    trg = trg.to(CFG.device)

    if train_type == 'feature':
      feature_out = feature_model(ipt)
      with torch.no_grad():
        out = model(feature_out)
    
    else:
      with torch.no_grad():
        feature_out = feature_model(ipt) 
      out = model(feature_out)
    
    loss = criterion(out, trg)
    losses.append(loss.detach().cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end = time.time()
    if i % CFG.print_freq == 0:
       print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses)}")
  return losses, accs

def validate(model, feature_model, dataloader, criterion, CFG):
  model.eval()
  accs = []
  losses = []
  start = time.time()

  with torch.no_grad():
    for i, (ipt, trg) in enumerate(dataloader):
      ipt = ipt.to(CFG.device)
      trg = trg.to(CFG.device)
      if feature_model is not None:
        out = feature_model(ipt)
      else:
        out = model(feature_model(ipt))
      
      loss = criterion(out, trg)
      losses.append(loss.cpu())

      end = time.time()
      if i % CFG.print_freq == 0:
        print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses)}")

  return losses, accs

CFG = cfg()  

### Load data, split and initialize dataloaders ###
CFG = cfg()
dataset = CatalanDataset(pd.read_csv(CFG.data_path), person_sensitive=True)
train1, train2, val, test = convert_dataload(datasplit(dataset), regularized_training=True)

print(f"Males train data size: {len(train1.dataset.data_table)}")
print(f"Females train data size: {len(train2.dataset.data_table)}")

#train1, train2, val, test = convert_dataload(datasplit(dataset), regularized_training=True)
#print(len(train1.dataset))
#print(len(train2.dataset)) 

if __name__ == 'main':
  main()