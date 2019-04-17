import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,info):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(info['input_size'], int(info['output_size']*info['output_size']))
        # self.fc2 = nn.Linear(500,)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(3, 16, kernel_size=3, stride=1,padding=1)
        self.convn = nn.Conv2d(16, 16, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, stride=1,padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(16, affine=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = x.reshape(1,1,200,200)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.relu(self.convn(x))
        x = self.bn1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x