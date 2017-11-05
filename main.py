import optparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import model
import batcher
import data
import os
import numpy as np

optparser = optparse.OptionParser()
optparser.add_option(
    "-c", "--emb_size", default="300",
    type='int', help="word embedding dimension"
)
optparser.add_option(
    "-v", "--vocab_size", default="2000000",
    type='long', help="vocabulary size"
)
optparser.add_option(
    "-C", "--hidden_size", default="150",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-l", "--layers", default="2",
    type='int', help="number of layers of GRU"
)
optparser.add_option(
    "-b", "--batch_size", default="4",
    type='int', help="batch_size"
)
optparser.add_option(
    "-g", "--use_cuda", default="1",
    type='int', help="whether use gpu"
)
optparser.add_option(
    "-d", "--dropout", default="0.5",
    type='float', help="dropout ratio"
)
optparser.add_option(
    "--lr", default="3e-2",
    type='float', help="learning rate"
)
optparser.add_option(
    "-s", "--seed", default="26",
    type='int', help="random seed for CPU and GPU"
)
optparser.add_option(
    "-i", "--iter", default="5000",
    type='long', help="num of iteration"
)
optparser.add_option(
    "--clip", default="3",
    type='float', help="result file location"
)
optparser.add_option(
    "-w", "--decay", default="0.0001",
    type='float', help="weight decay"
)
optparser.add_option(
    "-m", "--margin", default="16",
    type='int', help="margin of ranking loss"
)


def start_train(model, opts, wq_data_file):
    """
    Training the model
    """
    if opts.use_cuda == 1:
        print "Find GPU enable, using GPU to compute..."
        model.cuda()
        torch.cuda.manual_seed(opts.seed)
    else:
        print "Find GPU unable, using CPU to compute..."

    """create batcher"""
    wq_batcher = batcher.wq_batcher(opts, wq_data_file)

    # opt = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.decay)
    hidden = model.init_hidden()
    opt = optim.Adam(model.parameters(), lr=opts.lr)

    # if there exists a saved model, read it
    if os.path.isfile("checkpoint.tar"):
        print "found saved model, hold on, loading...."
        model.load_state_dict(torch.load("checkpoint.pkl"))
    # print h
    # delete cuda for local debug
    target = Variable(torch.ones(1)).cuda()
    cos_dist = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    MRL = torch.nn.MarginRankingLoss(margin=opts.margin, size_average=True)
    print "Training ..."
    model.train()

    for step in range(opts.iter):
        print '\n' + 'Epoch {} / {}'.format(step, opts.iter)
        print '-' * 10

        '''read next batch'''
        data = wq_batcher.next()

        '''choose the max cos similarity of positive relation and negative relation'''
        pr_max, nr_max = choose_highest_pr_nr(data, model, hidden, cos_dist, opts, wq_batcher)

        '''loss'''
        loss = MRL(pr_max, nr_max, target)

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), opts.clip)
        opt.step()
        print "-----Loss: ", loss.cpu().data[0]
    # print h
    print "hold on, saving model"
    torch.save(model.state_dict(), 'checkpoint.pkl')

    """evaluate"""
    evaluate(model, hidden, opts, wq_batcher, cos_dist)


def evaluate(model, hidden, opts, wq_batcher, cos_dist):
    model.eval()
    test_data = np.load("wq/wq_test_data.npy")
    total = 0
    res = []
    for i in range(len(test_data)):
        each = test_data[i]
        q, pr, nr = each[0], each[1], each[2]
        q = q * opts.batch_size
        q = np.reshape(q, (opts.batch_size, -1))
        q_tensor = Variable(torch.LongTensor(q)).cuda()
        # print "q_tensor size: ", q_tensor.size()
        h_q = model(q_tensor, hidden)

        # print "***after add h_q size: ", h_q.size()
        relations = pr + nr
        for j in range(len(relations) / opts.batch_size):
            r = relations[j * opts.batch_size: (j + 1) * opts.batch_size]
            while len(r) != opts.batch_size:
                r.append([0])
            try:
                r = wq_batcher.padding(r)
            except ValueError:
                print ValueError
                continue
            r_tensor = Variable(torch.LongTensor(r)).cuda()  # positive relation
            # print "relation_tensor size: ", r_tensor.size()
            h_r = model(r_tensor, hidden)

            tmp = cos_dist(h_q, h_r)
            res.append(tmp)
        res = torch.cat((res))
        # print "res size: ", res.size()
        value, index = torch.max(res, 0)
        print "****Predicted relation's index: ", index.data.cpu().numpy(), " ****Cosine relation max: ", value.data.cpu().numpy()
        if index.data.cpu().numpy() < len(pr):
            total += 1
        if i == 0:
            continue
        print "-------------Correct Prediction: {} / {}".format(total, i), "   Accumulate Accuracy: ", float(
            total / float(i))


def choose_highest_pr_nr(data, model, hidden, cos_dist, opts, wq_batcher):
    q, pr, nr = data[0], data[1], data[2]
    pr_cos, nr_cos = [], []

    '''question'''
    q = q * opts.batch_size
    q = np.reshape(q, (opts.batch_size, -1))
    q_tensor = Variable(torch.LongTensor(q)).cuda()
    h_q = model(q_tensor, q_tensor, hidden)

    for j in xrange(len(pr) / opts.batch_size + 1):
        r = pr[j * opts.batch_size: (j + 1) * opts.batch_size]
        while len(r) < opts.batch_size:
            r.append([0])
        try:
            r = wq_batcher.padding(r)
        except ValueError:
            print ValueError
            continue
        pr_tensor = Variable(torch.LongTensor(r)).cuda()  # positive relation
        # print "relation_tensor size: ", r_tensor.size()
        h_pr = model(q_tensor, pr_tensor, hidden)

        cos = cos_dist(h_q, h_pr)
        pr_cos.append(cos)

    pr_cos = torch.cat((pr_cos))
    pr_max, pr_maxindex = torch.max(pr_cos, 0)

    for j in range(len(nr) / opts.batch_size + 1):
        r = nr[j * opts.batch_size: (j + 1) * opts.batch_size]
        while len(r) < opts.batch_size:
            r.append([0])
        try:
            r = wq_batcher.padding(r)
        except ValueError:
            print ValueError
            continue
        nr_tensor = Variable(torch.LongTensor(r)).cuda()  # positive relation
        # print "relation_tensor size: ", r_tensor.size()
        h_nr = model(q_tensor, nr_tensor, hidden)

        cos = cos_dist(h_q, h_nr)
        nr_cos.append(cos)

    nr_cos = torch.cat((nr_cos))
    nr_max, nr_maxindex = torch.max(nr_cos, 0)

    return pr_max, nr_max

def main():
    # Read parameters from command line
    (opts, args) = optparser.parse_args()

    if not torch.cuda.is_available():
        opts.use_cuda = 1
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    md = model.BIGRU(opts)
    start_train(md, opts, "wq/wq_train_data.npy")


if __name__ == '__main__':
    main()
