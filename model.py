import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import optparse
import torch.optim as optim
import data
import torchvision.datasets as dsets
import torchvision.transforms as transforms

""" Bi-directional GRU-based RNN with Dynamic le`ngth"""

"""!!!!!!!!!NOT FINISH YET!!!!!!!!!!!"""

class BIGRU(nn.Module):
    """
    The RNN model for question representation
    """
    def __init__(self, opts):
        super(BIGRU, self).__init__()
        self._opts = opts
        self._batch_size = self._opts.batch_size
        # loader = data.loader()
        # self.vocab = loader.load_pickle("glove/smaller_glove_vocab.pickle")
        self.vec = torch.load("glove/smaller_glove_vec")

        self.word_embeddings = nn.Embedding(self._opts.vocab_size, self._opts.emb_size)
        self.word_embeddings.weight = nn.Parameter(self.vec)
        # self.word_embeddings.weight.data.copy_(torch.from_numpy(self.vec))

        self.gru = nn.GRU(self._opts.emb_size, self._opts.hidden_size, batch_first=True, dropout=self._opts.dropout, bidirectional=True)

        # self.fc_h = nn.Linear(2 * self._opts.hidden_size, self._opts.emb_size)

        self.dropout = nn.Dropout(p=self._opts.dropout)


    def forward(self, sentence, hidden):
        """
        Forward pass
        """

        input = self.word_embeddings(sentence)

        output, h = self.gru(input, hidden)

        h_dropout = self.dropout(h)

        # output_fc = self.fc_h(h_dropout.view(self._batch_size, -1))

        return output, h_dropout

    def init_hidden(self):
        """
        Initial the hidden state
        """
        h0 = Variable(torch.randn(2, self._batch_size, self._opts.hidden_size), requires_grad=False)
        # c0 = Variable(torch.randn(2, self._batch_size, self._opts.hidden_size), requires_grad=False)

        if self._opts.use_cuda:
            h0 = h0.cuda()
        return h0
