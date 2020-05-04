import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Net_LSTM(nn.Module):
    def __init__(self, vocabulary, dataset_lengths):
        super(Net_LSTM, self).__init__()
        # define layers
        self.lstm = nn.LSTM(input_size=1, hidden_size=200, num_layers=2, batch_first=True)
        self.word_embedding = nn.Embedding(num_embeddings=len(vocabulary), embedding_dim=1)
        self.hidden = torch.randn(2,42069,200)
        self.dataset_lengths = dataset_lengths



    def forward(self, input_list, dataset_lengths):


        X = self.word_embedding(input_list)  # sloppy cause I'm lazy
        X = torch.nn.utils.rnn.pack_padded_sequence(X, dataset_lengths, batch_first=True)
        X, hidden = self.lstm(X, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        return X, hidden

###########################



class Net_LSTM_drop(nn.Module):
    def __init__(self):
        super(Net_LSTM_drop, self).__init__()
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
        super(Net_GRU, self).__init__()
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
        super(Net_GRU_drop, self).__init__()
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
