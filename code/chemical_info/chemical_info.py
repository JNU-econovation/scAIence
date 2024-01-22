import torch
import torch.nn as nn
import torch.nn.functional as F


class ChemicalInfo(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super(ChemicalInfo, self).__init__()
        nn.Module.__init__(self)
        
        self.fc1 = nn.Linear(input_dim, 50)
        self.bn1 = nn.BatchNorm1d(50)

        self.fc2 = nn.Linear(50, 40)
        self.bn2 = nn.BatchNorm1d(40)
        
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(40, 32)
        self.bn3 = nn.BatchNorm1d(32)   
        
        self.dropout3 = nn.Dropout(0.2)  
          
        self.fc4 = nn.Linear(32, 20)
        self.bn4 = nn.BatchNorm1d(20)  
        
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(20, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)

        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        
        x = self.dropout4(x)
        
        x = self.fc5(x)
        
        return x