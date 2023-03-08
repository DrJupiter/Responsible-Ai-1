import torch
import torch.nn as nn

class FeatureModel(nn.Module):

    def __init__(self,input_shape):
        super(FeatureModel,self).__init__()
        self.fc1 = nn.Linear(input_shape,64)
        self.fc2 = nn.Linear(64,input_shape)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x