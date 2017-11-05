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
        self.max_length = 15
        # loader = data.loader()
        # self.vocab = loader.load_pickle("glove/smaller_glove_vocab.pickle")
        self.vec = torch.load("glove/smaller_glove_vec")

        self.word_embeddings = nn.Embedding(self._opts.vocab_size, self._opts.emb_size)
        self.word_embeddings.weight = nn.Parameter(self.vec)

        self.linear_q = nn.Linear(self._opts.hidden_size * 2, self._opts.hidden_size)
        self.linear_u = nn.Linear(self._opts.emb_size, self._opts.hidden_size)
        self.linear_h = nn.Linear(self._opts.hidden_size, self._opts.hidden_size)
        self.linear_s = nn.Linear(self._opts.hidden_size, 1)
        # self.gru = nn.GRU(self._opts.emb_size, self._opts.hidden_size, batch_first=True, dropout=self._opts.dropout, bidirectional=True)

        self.GRUCell = nn.GRUCell(self._opts.emb_size, self._opts.hidden_size)
        self.softmax = nn.Softmax()
        # input: batch(4) * length * 300
        # self.fc_h = nn.Linear(2 * self._opts.hidden_size, self._opts.emb_size)

        self.dropout = nn.Dropout(p=self._opts.dropout)


    def forward(self, question, relation, initial_hidden):
        """
        Forward pass
        """
        question_representation= []
        question = self.word_embeddings(question)
        relation = self.word_embeddings(relation)
        print "question embedding:", question.size()
        print "relation embedding:", relation.size()
        # quesion: batch(4) * length * 300
        # get the question representation
        hidden_f, hidden_b = initial_hidden, initial_hidden
        for i in xrange(question.size()[1]):
            hidden_f = self.GRUCell(question[:, i], hidden_f)
            hidden_b = self.GRUCell(question[:, question.size()[1] - i -1], hidden_b)
            print "hidden_f size:", hidden_f.size()
            # hidden state size: batch size * hidden size
            question_representation.append(torch.cat((hidden_f, hidden_b), dim=1).unsqueeze(1))

        hidden = question_representation[-1].squeeze(1)
        hidden = self.linear_q(hidden)
        question_representation = torch.cat(question_representation, dim=1)
        print "question representation:", question_representation.size()
        '''the initial hidden state of AttenGRU is the hidden state of the last time step of BiGRU'''

        print " hidden size:", hidden.size()

        for t in xrange(relation.size()[1]):
            st = []

            for j in xrange(question.size()[1]):
                s1 = self.linear_q(question_representation[:, j].squeeze(1))
                print "s1:", s1.size()
                s2 = self.linear_u(relation[:, t])
                print "s2:", s2.size()
                s3 = self.linear_h(hidden)
                print "s3:", s3.size()
                s = F.tanh(s1 + s2 + s3)
                print "s size:", s.size()
                s = self.linear_s(s)

                st.append(s)
            st = torch.cat(st, dim=1)
            print "before softmax st size:", st.size()
            at = self.softmax(st).unsqueeze(1)
            print "at size:", at.size()

            ct = torch.bmm(at, question_representation).squeeze(1)
            print "ct size:", ct.size()
            print "hidden size:", hidden.size()
            hidden = self.GRUCell(ct, hidden)
            print "after GRU hidden size:", hidden.size()

        return hidden
        # # output, hidden = self.gru(input, hidden)
        # print "{1st} Output:", output.size(), " {1st} hidden:", hidden.size()
        #
        # for i in xrange(input.size()[1]):
        #     current_word = input[:, i]
        #     q_feature = F.tanh(torch.cat((self.input_q(hidden), current_word)))
        #
        #
        # # output: batch(4) * length * 300; hidden: bidirectional&(2) * batch(4) * hidden size(300)
        # h_dropout = self.dropout(hidden)

        # output_fc = self.fc_h(h_dropout.view(self._batch_size, -1))

        # return output, h_dropout

    def init_hidden(self):
        """
        Initial the hidden state
        """
        h0 = Variable(torch.randn(self._batch_size, self._opts.hidden_size), requires_grad=False)

        if self._opts.use_cuda:
            h0 = h0.cuda()
        return h0
