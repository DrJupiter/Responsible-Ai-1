from model import Net_Logistic
from feature_model import FeatureModel
import torch
import torch.nn as nn
from dataload import CatalanDataset, datasplit, convert_dataload, dataloader_to_dataframe, get_encoding_table, group_split
import numpy as np
import pandas as pd
import random
import time 
from fairnessmetrics import test_fairness
from torch.utils.data import Dataset, DataLoader

# Reproduceability
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
      self.n_epochs = 50
      self.print_freq = 10000

def jointLoss(joint_preds, g1_preds, g2_preds, trg):
  group_criterion = torch.log.nn.BCELoss()

def main():
  CFG = cfg()  

  ### Load data, split and initialize dataloaders ###
  # dataset = CatalanDataset(pd.read_csv(CFG.data_path))
  # DataloaderTrain, DataloaderTrainG1, DataloaderTrainG2, DataloaderVal, DataloaderTest = convert_dataload(datasplit(dataset))

  CFG = cfg()
  enc_table = get_encoding_table()
  dataset = CatalanDataset(pd.read_csv(CFG.data_path), person_sensitive=True)
  trainingset, val, test = convert_dataload(datasplit(dataset))
  _x, _y = next(iter(trainingset))
  train_m, train_f = group_split(trainingset, enc_table, 'V1_sex')
  
  train_m = DataLoader(train_m, batch_size=24)
  train_f = DataLoader(train_f, batch_size=24)

  # print(f"Males train data size: {len(train_m)}")
  # print(f"Females train data size: {len(train_f)}")

  # TODO: Specify which groups you are splitting over and create a corresponding number of models and then loop over them

  ### Initialize model, optimizer and loss 
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
  criterion_m = nn.BCELoss(reduction='mean').to(CFG.device)
  criterion_n = nn.BCELoss(reduction='none').to(CFG.device)

  def Ld(out, target):
    
    """
    Original Ld code 1/1 from paper: (does not work)
    #L = torch.sum(torch.log(criterion_n(out, target)+1e-16))
    #return L
    """
    return torch.log(criterion_n(out, target))
    

  def L0(out,target):
    L = criterion_m(out, target) 
    return L

  def joint_loss(out_T, out_g1, out_g2, target): # Out tuple(torch.Tensor,torch.Tensor), got error if it was in there
    #LR = torch.abs(torch.mean(torch.log(criterion_n(out_g1, target)+1e-16) - torch.log(criterion_n(out_g2, target)+1e-16),dim=0))
    #LR = torch.mean(torch.log(out_g1) - torch.log(out_g2 + 1e-16),dim=0)
    LR = torch.abs(torch.log(criterion_m(out_g1, target)) - torch.log(criterion_m(out_g2, target)))
    L_0 = L0(out_T, target)
    return L_0 + 0.7*LR

  ### Start training loop ###
  for i in range(CFG.start_epoch, CFG.n_epochs):
    
    print("Training for group 1") #train_m
    train_loss_g1, train_acc_g1 = train(model_g1, feature_model,None, None, train_m, optimizer_g1, Ld, CFG, train_type='group')
    train_losses_g1.append(train_loss_g1)
    train_accs_g1.append(train_acc_g1)

    val_loss_g1, val_acc_g1 = validate(model_g1,feature_model, val, Ld, CFG)
    val_losses_g1.append(val_loss_g1)
    val_accs_g1.append(val_acc_g1)

    print("Training for group 2")
    train_loss_g2, train_acc_g2 = train(model_g2, feature_model,None, None, train_f, optimizer_g2, Ld, CFG, train_type='group')
    train_losses_g2.append(train_loss_g2)
    train_accs_g2.append(train_acc_g2)

    val_loss_g2, val_acc_g2 = validate(model_g2, feature_model, val, Ld, CFG)
    val_losses_g2.append(val_loss_g2)
    val_accs_g2.append(val_acc_g2)

    print("Training joint & feature model")
    train_loss, train_acc = train(joint_model, feature_model, model_g1, model_g2, trainingset, (feature_optimizer, joint_optimizer), (joint_loss, L0), CFG, train_type='feature')
    joint_train_losses.append(train_loss)
    joint_train_accs.append(train_acc)

    joint_val_loss, joint_val_acc = validate(joint_model, feature_model, val, L0, CFG)
    joint_val_losses.append(joint_val_loss)
    joint_train_accs.append(joint_val_acc)

  print("Testing final model")
  tst_loss, tst_acc = validate(joint_model, feature_model, test, L0, CFG)

  fairness_2(joint_model,feature_model, test, CFG)

  return joint_train_losses, joint_train_accs, joint_val_losses, joint_val_accs#, tst_loss, tst_acc


def train(model, feature_model, model_g1, model_g2, dataloader, optimizer, criterion, CFG, train_type='feature'):
  model.train()

  losses = []
  start = time.time()
  acc = 0

  for i, (ipt, trg) in enumerate(dataloader):
    ipt = ipt.to(CFG.device)
    trg = trg.to(CFG.device)

    if train_type == 'feature':
      feature_model.train()
      model_g1.eval()
      model.g2.eval()

      feature_out = feature_model(ipt)
      out = model(feature_out)
      #print(f"out: {out}")
      with torch.no_grad():
        
        out_g1 = model_g1(feature_out)
        out_g2 = model_g2(feature_out)
        
      #print(f"Out_g1: {out_g1.squeeze(1)}")
      #print(f"Out_g2: {out_g2.squeeze(1)}")'

      loss_f = criterion[0](out, out_g1, out_g2, trg)
      loss_j = criterion[1](out, trg)
      #losses.append((loss_f.detach().cpu(), loss_j.detach().cpu()))
      losses.append(loss_j.detach().cpu())
      optimizer[0].zero_grad()
      optimizer[1].zero_grad()

      loss_f.backward(retain_graph=True)
      loss_j.backward(retain_graph=True)

      optimizer[0].step()
      optimizer[1].step()
    
    else:
      feature_model.eval()
      with torch.no_grad():
        feature_out = feature_model(ipt)
        
      
      out = model(feature_out).squeeze(1)
      loss = criterion(out, trg)
      losses.append(loss.detach().cpu())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      #print("OUTTTT", out)

    acc += accuracy(out.round().reshape(-1).detach().cpu(), trg)
      
    end = time.time()
    #if i % CFG.print_freq == 0:
    #   print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses)}\nAcc: {acc / (trg.size(0)*i+1):.4f}")

  acc = acc/len(dataloader.dataset)
  print(f"FINAL ACCURACY: {acc:.4f}\nFinal loss: {np.mean(losses):.4f}")
  return losses, acc

def validate(model, feature_model, dataloader, criterion, CFG):
  model.eval()
  acc = 0
  losses = []
  start = time.time()

  with torch.no_grad():
    for i, (ipt, trg) in enumerate(dataloader):
      ipt = ipt.to(CFG.device)
      trg = trg.to(CFG.device)
      out = model(feature_model(ipt))
      
      loss = criterion(out, trg)
      losses.append(loss.cpu())

      acc += accuracy(out.round().reshape(-1).detach().cpu(), trg)
      end = time.time()
      #if i % CFG.print_freq == 0:
       # print(f"Time elapsed: {(end-start)/60:.4f} min\nAvg loss: {np.mean(losses)}")
  print(f"Final validation acc: {acc/len(dataloader.dataset):.4f}")
  print(f"Final validation loss: {np.mean(losses):.4f}")

  return losses, acc

def fairness_2(joint_model, feature_model, dataloader, CFG):
  joint_model.eval()
  feature_model.eval()
  predictions = []
  
  with torch.no_grad():
    for i, (ipt, trg) in enumerate(dataloader):
      ipt = ipt.to(CFG.device)
      trg = trg.to(CFG.device)
      
      #np.where(model(ipt).detach().cpu().numpy() <= 0.5, 1, 0)
      predictions.append((joint_model(feature_model(ipt)).round()).reshape(-1).detach().cpu().numpy())
   
  df = dataloader_to_dataframe(dataloader, dataloader.dataset.dataset.columns)

  test_fairness(df, predictions,save_path = "results/fairness_mitigation.json", fairness_test_group='V1_sex')

def accuracy(pred, trg):
  acc = 0
  for i in range(pred.size(0)):
    if pred[i]==trg[i]:
      acc+=1
  
  return acc
### Load data, split and initialize dataloaders ###


#train1, train2, val, test = convert_dataload(datasplit(dataset), regularized_training=True)
#print(len(train1.dataset))
#print(len(train2.dataset)) 

# if __name__ == 'main':
main()