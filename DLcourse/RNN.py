import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""""
from keras.models import Sequential
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(1,input_shape=(1,200),return_sequences=True))
model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
"""""

class Net_LSTM(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()
        # define layers
        self.lstm = nn.LSTM(200,200, dropout=0)
        self.lstm_drop = nn.LSTM(200, 200, dropout=1)
        self.gru = nn.GRU(200,200 , dropout=0)
        self.gru_drop = nn.GRU(200, 200, dropout=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(12)
        self.dropout = nn.Dropout2d(0.25)

  # define forward function
    def forward(self, t):
        # conv 1
        t = self.lstm(t)

        # output
        #t = F.softmax(t)
        #t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t

###########################
model = Net_LSTM()
model.parameters()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
params

class Net_Dropout(nn.Module):
    def __init__(self):
        super(Net_Dropout, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        #self.batchnorm1 = nn.BatchNorm2d(6)
        #self.batchnorm2 = nn.BatchNorm2d(12)
        self.dropout = nn.Dropout2d(0.25)

  # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        #t = self.batchnorm1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.dropout(t)

        # conv 2
        t = self.conv2(t)
        #t = self.batchnorm2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.dropout(t)

        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t

######################

class Net_BatchNorm(nn.Module):
    def __init__(self):
        super(Net_BatchNorm, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(12)
        #self.dropout = nn.Dropout2d(0.25)

  # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = self.batchnorm1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #t = self.dropout(t)

        # conv 2
        t = self.conv2(t)
        t = self.batchnorm2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #t = self.dropout(t)

        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t

######################

class Net_None(nn.Module):
    def __init__(self):
        super(Net_None, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        #self.batchnorm1 = nn.BatchNorm2d(6)
        #self.batchnorm2 = nn.BatchNorm2d(12)
        #self.dropout = nn.Dropout2d(0.25)

    # define forward function
    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        # t = self.batchnorm1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #t = self.dropout(t)

        # conv 2
        t = self.conv2(t)
        # t = self.batchnorm2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #t = self.dropout(t)

        # fc1
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t