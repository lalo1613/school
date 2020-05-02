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
        self.lstm_drop = nn.LSTM(200,hidden_size=2, dropout=0.5)
        self.gru = nn.GRU(200,200 , dropout=0)
        self.gru_drop = nn.GRU(200, 200, dropout=0.5)
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

class Net_LSTM_drop(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()
        # define layers
        self.lstm = nn.LSTM(200,200, dropout=0)
        self.lstm_drop = nn.LSTM(200,hidden_size=2, dropout=0.5)
        self.gru = nn.GRU(200,200 , dropout=0)
        self.gru_drop = nn.GRU(200, 200, dropout=0.5)
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
        t = self.lstm_drop(t)

        # output
        #t = F.softmax(t)
        #t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t

###########################


class Net_GRU(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()
        # define layers
        self.lstm = nn.LSTM(200,200, dropout=0)
        self.lstm_drop = nn.LSTM(200,hidden_size=2, dropout=0.5)
        self.gru = nn.GRU(200,200 , dropout=0)
        self.gru_drop = nn.GRU(200, 200, dropout=0.5)
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
        t = self.gru(t)

        # output
        #t = F.softmax(t)
        #t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t

###########################

class Net_GRU_drop(nn.Module):
    def __init__(self):
        super(Net_LSTM, self).__init__()
        # define layers
        self.lstm = nn.LSTM(200,200, dropout=0)
        self.lstm_drop = nn.LSTM(200,hidden_size=2, dropout=0.5)
        self.gru = nn.GRU(200,200 , dropout=0)
        self.gru_drop = nn.GRU(200, 200, dropout=0.5)
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
        t = self.gru_drop(t)

        # output
        #t = F.softmax(t)
        #t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.

        return t
