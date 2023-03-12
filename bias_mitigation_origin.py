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
      self.n_epochs = 10
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
  training_set, validation_set, test_set = convert_dataload(datasplit(dataset))
  _x, _y = next(iter(training_set))
  #train_g1, train_g2, train_g3, train_g4, train_g5 = group_split(trainingset, enc_table, 'V4_area_origin')
  group='V4_area_origin'
  group_splits = group_split(training_set, enc_table, 'V4_area_origin')
  group_splits = convert_dataload(group_splits, batchsizes=[24]*len(group_splits), shuffle=[True]*len(group_splits)) 


  # print(f"Males train data size: {len(train_m)}")
  # print(f"Females train data size: {len(train_f)}")

  # TODO: Specify which groups you are splitting over and create a corresponding number of models and then loop over them

  ### Initialize model, optimizer and loss 
  joint_model = Net_Logistic(_x.size()[1]).to(CFG.device)
  feature_model = FeatureModel(_x.size()[1]).to(CFG.device)
  group_models = [Net_Logistic(_x.size()[1]).to(CFG.device) for _ in range(len(group_splits))]

  joint_optimizer = torch.optim.Adam(joint_model.parameters(), lr=CFG.lr)
  feature_optimizer = torch.optim.Adam(feature_model.parameters(), lr=CFG.lr)
  group_optimizers = [torch.optim.Adam(group_model.parameters(), lr=CFG.lr) for group_model in group_models]

  joint_train_losses, joint_val_losses, joint_train_accs, joint_val_accs = [], [], [], []
  feature_train_losses, feature_val_losses, feature_train_accs, feature_val_accs = [], [], [], []
  group_log = {i: {'train_loss': [], 'validation_loss': [], 'train_accuracy': [], 'validation_accuracy': [] } for i in range(len(group_splits))}

  ### Define custom loss functions ###
  criterion_m = nn.BCELoss(reduction='mean').to(CFG.device)
  criterion_n = nn.BCELoss(reduction='none').to(CFG.device)

  def Ld(out, target):
    
    """
    Original Ld code 1/1 from paper: (does not work)
    #L = torch.sum(torch.log(criterion_n(out, target)+1e-16))
    #return L
    """
    return torch.log(criterion_m(out, target))

  def L0(out,target):
    L = criterion_m(out, target) 
    return L

  def joint_loss(out_T, group_out, target): # Out tuple(torch.Tensor,torch.Tensor), got error if it was in there
    #LR = torch.abs(torch.mean(torch.log(criterion_n(out_g1, target)+1e-16) - torch.log(criterion_n(out_g2, target)+1e-16),dim=0))
    #LR = torch.mean(torch.log(out_g1) - torch.log(out_g2 + 1e-16),dim=0)

    LR = 0
    for out in group_out:
      LR += torch.log(criterion_m(out, target))
    LR = torch.mean(LR)
    L_0 = L0(out_T, target)
    return L_0 + 0.7*LR
  
  ### Start training loop ###
  for i in range(CFG.start_epoch, CFG.n_epochs):    

    for (g, (group_split, group_model, group_optimizer)) in enumerate(zip(group_splits, group_models, group_optimizers)):
      # TODO: checkout criterion here
      print(f"Training group {g}") # TODO: Print which group this is semantically
      loss, accuracy = train(group_model, feature_model, None, group_split, group_optimizer, criterion=Ld, CFG=CFG, train_type='group') 


      group_log[g]['train_loss'].append(loss)
      group_log[g]['train_accuracy'].append(accuracy)

      # TODO: Potential bug in the reusing of the validation set
      loss, accuracy = validate(group_model, feature_model, validation_set, Ld, CFG)
      group_log[g]['validation_loss'].append(loss)
      group_log[g]['validation_accuracy'].append(accuracy)



    
    print("Training joint & feature model")
    train_loss, train_acc = train(joint_model, feature_model, group_models , training_set, (feature_optimizer, joint_optimizer), (joint_loss, L0), CFG, train_type='feature')
    joint_train_losses.append(train_loss)
    joint_train_accs.append(train_acc)

    joint_val_loss, joint_val_acc = validate(joint_model, feature_model, validation_set, L0, CFG)
    joint_val_losses.append(joint_val_loss)
    joint_train_accs.append(joint_val_acc)

  print("Testing final model")
  tst_loss, tst_acc = validate(joint_model, feature_model, test_set, L0, CFG)

  fairness_2(joint_model,feature_model, test_set, CFG)

  return joint_train_losses, joint_train_accs, joint_val_losses, joint_val_accs#, tst_loss, tst_acc


def train(model, feature_model, group_models, dataloader, optimizer, criterion, CFG, train_type='feature'):

  losses = []
  start = time.time()
  acc = 0

  model.train()

  for i, (ipt, trg) in enumerate(dataloader):
    ipt = ipt.to(CFG.device)
    trg = trg.to(CFG.device)

    if train_type='group':
      feature_model.eval()
      with torch.no_grad():
        feature_out = feature_model(ipt)
        

      out = model(feature_out).squeeze(1)
      loss = criterion(out, trg)
      losses.append(loss.detach().cpu())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    elif train_type == 'feature':
      feature_model.train()
      feature_out = feature_model(ipt)
      out = model(feature_out)
      #print(f"out: {out}")
      with torch.no_grad():
        # TODO: Potential error here attempt copy
        for group_model in group_models:
          group_model.eval()

        group_outs = [group_model(feature_out) for group_model in group_models]


      loss_f = criterion[0](out, group_outs, trg)
      loss_j = criterion[1](out, trg)

      losses.append(loss_j.detach().cpu())
      optimizer[0].zero_grad()
      optimizer[1].zero_grad()

      loss_f.backward(retain_graph=True)
      loss_j.backward(retain_graph=True)

      optimizer[0].step()
      optimizer[1].step()
    


    # TODO: Double check this makes sense
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

  test_fairness(df, predictions,save_path = "results/fairness_mitigation.json", fairness_test_group='V4_area_origin')

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