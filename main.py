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
    "-C", "--hidden_size", default="50",
    type='int', help="Char LSTM hidden layer size"
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
    "--lr", default="3e-3",
    type='float', help="learning rate"
)
optparser.add_option(
    "-s", "--seed", default="26",
    type='int', help="random seed for CPU and GPU"
)
optparser.add_option(
    "-i", "--iter", default="4",
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
    h = model.init_hidden()
    opt = optim.Adam(model.parameters(), lr=opts.lr)

    # if there exists a saved model, read it
    if os.path.isfile("checkpoint.tar"):
        print "found saved model, hold on, loading...."
        model.load_state_dict(torch.load("checkpoint.pkl"))

    # delete cuda for local debug
    target = Variable(torch.ones(opts.batch_size)).cuda()
    cos_dist = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    MRL = torch.nn.MarginRankingLoss(margin=opts.margin, size_average=True)
    print "Training ..."
    model.train()
    # print h
    for step in range(opts.iter):
        print '\n' + 'Epoch {} /{}'.format(step, opts.iter)
        print '-' * 10

        # loss = 0
        # delete cuda for local debug
        q, pr, nr = wq_batcher.next()
        q_tensor = Variable(torch.LongTensor(q)).cuda()
        pr_tensor = Variable(torch.LongTensor(pr)).cuda()  # positive relation
        nr_tensor = Variable(torch.LongTensor(nr)).cuda()  # negative relation

        # print " original q_tensor size: ", q_tensor.size()
        # print "pr_tensor size: ", pr_tensor.size()
        # print "nr_tensor size: ", nr_tensor.size()

        y_q, h_q = model(q_tensor, h)
        y_pr, h_pr = model(pr_tensor, h)
        y_nr, h_nr = model(nr_tensor, h)

        # print "***y_q size: ", y_q.size()
        #
        # print "*** original h_q size: ", h_q.size()  # 2*batch size * 50
        # print "***h_pr size: ", h_pr.size()
        # print "***h_nr size: ", h_nr.size()

        h_q = h_q[0] + h_q[1]
        h_pr = h_pr[0] + h_pr[1]
        h_nr = h_nr[0] + h_nr[1]
        # print "*** after add h_q size: ", h_q.size()  # batch size
        # print "***h_pr size: ", h_pr.size()
        # print "***h_nr size: ", h_nr.size()
        input1 = cos_dist(h_q, h_pr)
        input2 = cos_dist(h_q, h_nr)

        # print "***cos distance size: ", input1.size()
        # print "***input2 size: ", input2.size()

        # print input1
        # print input2
        # print target
        loss = MRL(input1, input2, target)
        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), opts.clip)
        opt.step()
        print "-----Loss: ", loss.cpu().data[0]
    # print h
    print "hold on, saving model"
    torch.save(model.state_dict(), 'checkpoint.pkl')

    """evaluate"""
    model.eval()
    test_data = np.load("wq/wq_test_data.npy")
    for i in range(len(test_data)):
        each = test_data[i]
        q, pr, nr = each[0], each[1], each[2]
        q = [q, q, q, q]
        q_tensor = Variable(torch.LongTensor(q)).cuda()
        # print "q_tensor size: ", q_tensor.size()
        y_q, h_q = model(q_tensor, h)
        h_q = h_q[0] + h_q[1]
        # print "***after add h_q size: ", h_q.size()
        total = 0
        relations = pr + nr
        for j in range(len(relations) / opts.batch_size):
            r = relations[j * opts.batch_size: (j + 1) * opts.batch_size]
            r = wq_batcher.padding(r)
            r_tensor = Variable(torch.LongTensor(r)).cuda()  # positive relation
            # print "relation_tensor size: ", r_tensor.size()
            y_r, h_r = model(r_tensor, h)
            h_r = h_r[0] + h_r[1]
            tmp = cos_dist(h_q, h_r)
            if j == 0:
                res = torch.cat((tmp, tmp))
            else:
                res = torch.cat((res, tmp))

        print "res size: ", res.size()
        value, index = torch.max(res, 0)
        print "index:", index, "max: ", value
        if index.data.cpu().numpy() < len(pr):
            total += 1
        if i == 0:
            continue
        print "----total_accuracy: ", float(total/i)


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
