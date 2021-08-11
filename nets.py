from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import TensorDataset
from torch.nn import functional as F
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from torch.nn import Conv2d

from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4



class GroupCNN(nn.Module):
    
    
    def __init__(self):
        super(GroupCNN, self).__init__()
        self.conv1 = P4ConvZ2(1, 16, kernel_size=3)
        self.conv2 = P4ConvP4(16, 32, kernel_size=3)
        self.conv3 = P4ConvP4(32, 12, kernel_size=3)
        self.conv4 = P4ConvP4(12, 8, kernel_size=3)
        self.fc1 = nn.Linear(800, 32)
        self.fc2 = nn.Linear(32, 10)


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        
        # now flatten and MLP
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training,)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    
    
class CNN(nn.Module):
    
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(1, 16, kernel_size=3)
        self.conv2 = Conv2d(16, 32, kernel_size=3)
        self.conv3 = Conv2d(32, 12, kernel_size=3)
        self.conv4 = Conv2d(12, 8, kernel_size=3)
        self.fc1 = nn.Linear(800, 32)
        self.fc2 = nn.Linear(32, 10)


    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        
        # now flatten and MLP
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training,)
        x = self.fc2(x)
        return F.log_softmax(x)
    







