import pandas as pd
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator
import numpy as np
from typing import List

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

def convert_dataload(array: List[Dataset], batchsizes=[24,1,1], shuffle=[True, False, False], regularized_training=False):
    assert len(array) == len(batchsizes),"Number of datasets must match batchsize setting length" 
    assert len(array) == len(shuffle), "Number of datasets must match shuffle setting length"
    if not regularized_training:
        return [DataLoader(dataset,batch_size=batchsize, shuffle=s) for (dataset, batchsize, s) in zip(array, batchsizes, shuffle)]
    
    else:
        # TODO fix shitty path bug, refactor dataload script with "split" column in df?
        train_df = array[0].dataset.data_table.copy()
        train_male = CatalanDataset(train_df.loc[train_df['V1_sex'] == 0], person_sensitive=True)
        train_female = CatalanDataset(train_df.loc[train_df['V1_sex'] == 1], person_sensitive=True)
        
        # split train into male and female...
        #train_male = CatalanDataset(array[0].dataset.data_table.loc[array[0].dataset.data_table['V1_sex'] == 0], person_sensitive=True)
        #train_female = CatalanDataset(array[0].dataset.data_table.loc[array[0].dataset.data_table['V1_sex'] == 1], person_sensitive=True)
        
        batchsizes = [batchsizes[0], batchsizes[0]] + batchsizes[1:]
        shuffle = [shuffle[0], shuffle[0]] + shuffle[1:]
        return [DataLoader(dataset,batch_size=batchsize, shuffle=s) for (dataset, batchsize, s) in zip(array, batchsizes, shuffle)]


class CatalanDataset(Dataset):

    def __init__(self, path, person_sensitive=False, regularized_training=False) -> None:
        super().__init__()
        self.data_table = pd.read_csv(path)
        self.data_table = self.data_table[self.data_table.columns[1:]]
        if not person_sensitive:
            self.data_table = self.data_table[self.data_table.columns[5:]]
    
        self.feature = self.data_table.loc[:, self.data_table.columns != 'V115_RECID2015_recid'].values.astype(np.float32)
        self.target = self.data_table.loc[:, self.data_table.columns == 'V115_RECID2015_recid'].values.astype(np.float32)
    
    def __getitem__(self, index):
        return self.feature[index],self.target[index]
    
    def __len__(self):
        return len(self.target)

        

        


if __name__ == "__main__":
    data_table = pd.read_csv('data/catalan-juvenile-recidivism-subset.csv')
    preprocess()
    dataset = CatalanDataset('./data/preprocessed.csv')
    train, validation, test = datasplit(dataset)
    print('succes')