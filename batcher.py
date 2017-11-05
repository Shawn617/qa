import json
from pprint import pprint
import pickle
import csv
import numpy as np
import torch.nn.functional as F
import data
import torch
import os
from random import choice


class wq_batcher(object):
    """
    Webquestion data load bathcer
    """
    def __init__(self, opts, wq_data_file):
        self._batch_size = opts.batch_size
        # self._loader = data.loader("glove/glove_vocab.pickle", "glove/glove_vec")
        self._wq_data = np.load(wq_data_file)
        self._data_size = len(self._wq_data)
        self.index = -1
        print "Create WebQuestion_Batcher"

    def next(self):
        '''
        return next question data
        :return:
        '''
        self.index += 1
        if self.index >= self._data_size:
            self.index = -1
        return self._wq_data[self.index]


    # def next(self):
    #     """
    #     :return: return next batch data
    #     random select positive relation and negative relation
    #     """
    #     q, pr, nr = [], [], []
    #     max_length = 0
    #     print "index: ", self.index
    #     for index in range(self.index, self.index + self._batch_size):
    #         each = self._wq_data[index]
    #         question, posi_relation, nega_relation = each[0], each[1], each[2]
    #         posi_relation = choice(posi_relation)
    #         nega_relation = choice(nega_relation)
    #         q.append(question)
    #         pr.append(posi_relation)
    #         nr.append(nega_relation)
    #     self.index += self._batch_size
    #     '''reset index if out of range'''
    #     if self.index + self._batch_size > self._data_size:
    #         self.index = 0
    #
    #     q = self.padding(q)
    #     pr = self.padding(pr)
    #     nr = self.padding(nr)
    #     return q, pr, nr


    def padding(self, input, maxlength=0):
        """
        padding the sequence to the longest seq
        :param input: question indexes matrix
        :param maxlength:
        :return:
        """
        temp = []
        maxlen = maxlength if maxlength != 0 else len(max(input, key=len))
        for i in xrange(len(input)):
            temp.append(input[i] + [0] * (maxlen - len(input[i])))
        return temp

    def get_all_predicate(self, subj_mid, subj_name):
        cmd = "mono /mnt/data/qile/FastRDFStore/FastRDFStoreClient/bin/Release/FastRDFStoreClient.exe -s deepc11.acis.ufl.edu -m " + subj_mid
        f = os.popen(cmd)
        output = f.read()
        f.close()
        if "type.object.name" in output:
            start, end= output.find("type.object.name") + 45, output.find("Took") - 1
        if output[start: end].lower() == subj_name.lower():
            print "subject entity found"
        else:
            print "subject entity not found"

        output = output[65: start - 46].split("\n")
        predicate = []
        for i in range(len(output)):
            line = output[i]
            line = line.split("-->")
            if line[0].lstrip().rstrip() == "":
                continue
            predicate.append(line[0].lstrip().rstrip())
        return predicate


class sq_batcher(object):
    """
    Webquestion data load bathcer
    """
    def __init__(self, batch_size, sq_pickle):
        self._batch_size = batch_size
        self._loader = data.loader("glove/glove_vocab.pickle", "glove/glove_vec")
        self._sq_dict = self._loader.load_pickle(sq_pickle)
        self.index = 0
        print "Create SimpleQuestion_Batcher"

    def next(self):
        """
        :return: return next batch data
        """
        sentences, subj_mid, predicate, obj_mid,  = [], [], [], []
        for key in self._sq_dict:
            line = self._sq_dict[key]
            sentences.append(line[0])
            subj_mid.append(line[1])
            self.get_all_predicate(line[1])
            obj_mid.append(line[3])
        return self.padding(sentences), subj_mid, predicate, obj_mid

    def padding(self, input, maxlength=0):
        """
        padding the sequence to the longest seq
        :param input: question indexes matrix
        :param maxlength:
        :return:
        """
        temp = []
        maxlen = maxlength if maxlength != 0 else len(max(input, key=len))
        for i in xrange(self._batch_size):
            temp.append(input[i] + [0] * (maxlen - len(input[i])))
        return temp

    def get_all_predicate(self, subj_mid):
        cmd = "mono /mnt/data/qile/FastRDFStore/FastRDFStoreClient/bin/Release/FastRDFStoreClient.exe -s deepc11.acis.ufl.edu -m " + subj_mid
        f = os.popen(cmd)
        output = f.read()
        f.close()

        output = output[65: output.find("type.object.name") - 1].split("\n")
        predicate = []
        for i in range(len(output)):
            line = output[i]
            line = line.split("-->")
            if line[0].lstrip().rstrip() == "":
                continue
            predicate.append(line[0].lstrip().rstrip())
        return predicate




"""test wq_batcher"""
# wq_batcher = wq_batcher(4, "wq/wq_train_data.npy")
# print wq_batcher.next()