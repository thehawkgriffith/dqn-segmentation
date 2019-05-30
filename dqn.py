import torch 
import torch.nn as nn
import numpy as np 

class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape[0]*input_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        inp_x = []
        if x.shape[0] != 4:
            for t in x:
                inp_x.append(t.view(1, -1))
            x = torch.tensor([t.cpu().numpy() for t in inp_x]).cuda()
        else:
            x = x.view(1, -1)
        return self.fc(x)