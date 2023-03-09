import pandas as pd
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator
import numpy as np
from typing import List
import torch

def preprocess(path='data/catalan-juvenile-recidivism-subset.csv', save_path='data/') -> None: 
    """
    Preprocess the catalan dataset 
    by converting text features to numeric embeddings
    """
    if os.path.isfile(path):
        data_table = pd.read_csv(path)
        encoding_dicts = {}
        for column in data_table.columns:
            if data_table[column].dtype == 'object':
                data_table[column], map_dict = encode(data_table[column])
                encoding_dicts.update({column: map_dict})

            data_table.to_csv(f'{save_path}preprocessed.csv', index=False)
            with open(f'{save_path}encoding.json', 'w') as json_file:
                json.dump(encoding_dicts,json_file)
                json_file.close()
    else:
        raise FileExistsError(f'Unable to find {path}')
    return None

def encode(data_frame):
    """
    Takes a data_frame and numerically encodes unique items
    with natural numbers
    """
    unique = pd.unique(data_frame) 
    map_dict = {item: i for i, item in enumerate(unique)}
    return data_frame.map(map_dict), map_dict

def datasplit(dataset, split=[0.3,0.3,0.4], r_idx=2, seed=42):
    """
    Splits the data into train, validation and test
    """

    train_size = int(len(dataset)*split[0])
    validation_size = int(len(dataset) * split[1])
    test_size = int(len(dataset) * split[2])
    sizes = [train_size, validation_size, test_size]
    r = len(dataset)-sum(sizes)
    sizes[r_idx] += r
    return random_split(dataset, sizes, Generator().manual_seed(seed))

def group_split(dataset, encoding_table, group):

    idx = None
    group_encoding = None 
    for i, key in enumerate(encoding_table.keys()):
        if group == key:
            idx = i
            group_encoding = {value: key for key, value in encoding[group].items()}

            break
    if idx is None:
        raise ValueError("Unable to find group")

    # Expected form (N, (feature,target))    
    realized_dataset = np.array(list(iter(dataset)),dtype=np.object)
    merged = np.array([np.concatenate((x,y)) for x,y in realized_dataset])
    unique = np.unique(merged[:,idx])

    group_splits = [] 
    for u in unique:
        group_splits.append(CatalanDatasetGroup(merged[np.where(merged[:,idx] == u)], group, group_encoding[u]))
    
    return group_splits

def get_encoding_table(path='./data/encoding.json'):
    with open(path,'r') as f:
        encoding_table = json.load(f)
        f.close()
    return encoding_table


def convert_dataload(array: List[Dataset], batchsizes=[24,1,1], shuffle=[True, False, False]):
    assert len(array) == len(batchsizes),"Number of datasets must match batchsize setting length" 
    assert len(array) == len(shuffle), "Number of datasets must match shuffle setting length"
    return [DataLoader(dataset,batch_size=batchsize, shuffle=s) for (dataset, batchsize, s) in zip(array, batchsizes, shuffle)]
  

class CatalanDatasetGroup(Dataset):

    def __init__(self, numpy_array, group, subgroup) -> None:
        super().__init__()
        self.dataset = numpy_array
        self.group = group
        self.subgroup = subgroup

    def __getitem__(self, index):
        return self.dataset[index,:-1],self.dataset[index,-1]
    
    def __len__(self):
        return len(self.dataset)

class CatalanDataset(Dataset):

    def __init__(self, data_table, person_sensitive=False, regularized_training=False) -> None:
        super().__init__()
        
        self.data_table = data_table[data_table.columns[1:]]
        if not person_sensitive:
            self.data_table = self.data_table[self.data_table.columns[5:]]
    
        self.feature = self.data_table.loc[:, self.data_table.columns != 'V115_RECID2015_recid'].values.astype(np.float32)
        self.target = self.data_table.loc[:, self.data_table.columns == 'V115_RECID2015_recid'].values.astype(np.float32)
    
    def __getitem__(self, index):
        return self.feature[index],self.target[index]
    
    def __len__(self):
        return len(self.target)
    

# TODO: FUNCTION FOR CONVERTING DATALOADER BACK TO PD DATAFRAME WITH ORIGINAL COLUMNS FOR FAIRNESS METRIC FUNCTIONS        
def dataloader_to_dataframe(dataloader, columns):
    """
    Requires columns to be in the correct order
    (In our case putting the target variable(s) last)
    """
    # realized_dataset = np.array(list(iter(dataloader)),dtype=np.object)
    # merged = np.array([np.concatenate((x,y)) for x,y in realized_dataset])
    # return pd.DataFrame(merged, columns=columns)
    realized_dataset = list(iter(dataloader))
    merged = np.array([np.array(torch.hstack((x[0],y[0]))) for x,y in realized_dataset])
    return pd.DataFrame(merged, columns=columns)


if __name__ == "__main__":
    data_table = pd.read_csv('data/catalan-juvenile-recidivism-subset.csv')
    preprocess()
    processed_data = pd.read_csv('./data/preprocessed.csv')
    dataset = CatalanDataset(processed_data)
    train, validation, test = datasplit(dataset)
    encoding = get_encoding_table()
    groups = group_split(train, encoding, 'V1_sex')
    print('succes')