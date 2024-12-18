import torch.nn as nn
import torch
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, Y, Z):
        input = torch.cat((Y, Z), dim=1)
        output = self.bn1(F.elu(self.fc1(input)))
        output = self.bn2(F.elu(self.fc2(output)))
        output = self.fc3(output)
        return output

    # def forward(self, x):
    #     return self.net(x)
