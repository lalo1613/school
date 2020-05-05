import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLSTM(nn.Module):
    def __init__(self, vocabulary, embd_dim, hidden_size):
        super(NetLSTM, self).__init__()
        # define layers
        self.lstm = nn.LSTM(input_size=embd_dim, hidden_size=hidden_size, num_layers=2)
        self.word_embedding = nn.Embedding(num_embeddings=len(vocabulary), embedding_dim=embd_dim)
        self.lin = nn.Linear(hidden_size, len(vocabulary))

    def forward(self, inputs, input_lengths, hidden):
        embedded_input = self.word_embedding(inputs)
        X = torch.nn.utils.rnn.pack_padded_sequence(embedded_input, input_lengths, batch_first=True)
        X, hidden = self.lstm(X)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = self.lin(X)
        log_probs = F.softmax(X, 1)

        return log_probs, hidden
