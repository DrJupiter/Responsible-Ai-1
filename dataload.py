import pandas as pd
import numpy as np
import os
import json

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

def datasplit(path='data/preprocessed.csv'):
    """
    Splits the data into train, validation and test
    """
    pass
    


if __name__ == "__main__":
    data_table = pd.read_csv('data/catalan-juvenile-recidivism-subset.csv')
    preprocess()
    print('succes')