import json
from pprint import pprint
import pickle
import csv
import numpy as np
import nltk
import torchwordemb as we
import torch

class loader(object):
    def __init__(self):
        # self.glove_vocab = self.load_pickle(vocab_file)
        # self.glove_vec = torch.load(vec_file)
        print "Create Loader"

    # def load_wq_json(self, wq_file, glove_vocab):
    #     """
    #         read raw webquestion json file, extract useful info and save them into a dict
    #         :param wq_file: input raw webquestion json file
    #         :return:
    #         """
    #     with open(wq_file) as wq:
    #         data = json.load(wq)
    #         data = data["Questions"]
    #
    #     wq_dict = {}
    #     for i in range(len(data)):
    #         q_id = data[i]["QuestionId"]
    #         Processed_Question = data[i]["ProcessedQuestion"] if data[i]["ProcessedQuestion"] != None else ""
    #         parse = data[i]["Parses"][0]
    #         # PotentialTopicEntityMention = parse["PotentialTopicEntityMention"]
    #         TopicEntityMid = parse["TopicEntityMid"] if parse["TopicEntityMid"] != None else ""
    #         TopicEntityName = parse["TopicEntityName"] if parse["TopicEntityName"] != None else ""
    #         if parse["Answers"] == []: # if there were no answer for a specific question, then skip it.
    #             AnswerArgument = ""
    #             EntityName = ""
    #         else:
    #             AnswerArgument = parse["Answers"][0]["AnswerArgument"]
    #             EntityName = parse["Answers"][0]["EntityName"]
    #         sent = Processed_Question.split()
    #         sent_index = []
    #         for word in sent:
    #             if word in glove_vocab:
    #                 sent_index.append(glove_vocab[word])
    #             else:
    #                 sent_index.append(0)
    #         tmp = [sent_index, TopicEntityMid, TopicEntityName, AnswerArgument, EntityName] # question, entity_id, entity_name, answer_id, answer_name
    #         wq_dict[str(q_id)] = tmp
    #     self.save_pickle(wq_dict, "wq/WebQSP.train.pickle")
    #     return len(wq_dict)

    def load_wq_relations(self, glove_vocab, wq_train_file, wq_test_file, relations_file, save_train_data, save_test_data, save_index2rela):
        # produce a dictionary mapping index with its relation/predicate
        wq_index2relations = [[]] #mapping index to relation
        with open(relations_file) as relations:
            for line in relations:
                tmp = line.replace("..", " ").replace("_", " ").replace(".", " ").replace("\n", "")
                tmp = tmp.split(" ")
                index_tmp = []
                for word in tmp:
                    if word in glove_vocab:
                        index_tmp.append(glove_vocab[word])
                    else:
                        index_tmp.append(0)
                wq_index2relations.append(index_tmp)
        print "wq_index2relations size:", len(wq_index2relations)
        np.save(save_index2rela, wq_index2relations)

        # lodaing webquestion training data
        wq_train_data = []
        with open(wq_train_file) as rela:
            for line in rela:
                q, pr, nr = [], [], []
                line = line.split("	")
                posi_rela, nega_rela, question = line[0], line[1], line[2]
                # print "posi_rela", posi_rela, "nega_rela", nega_rela, "quesiton", question

                # turn negative relations into indexes
                nega_rela = nega_rela.split(" ")
                for nega in nega_rela:
                    if int(nega) >= len(wq_index2relations):
                        print"***index:", int(nega)
                    relation = wq_index2relations[int(nega)]
                    nr.append(relation)
                # turn positive relations into indexes
                posi_rela = posi_rela.split(" ")
                for posi in posi_rela:
                    relation = wq_index2relations[int(posi)]
                    pr.append(relation)
                # turn question into indexes
                question = question.split(" ")
                for word in question:
                    if word in glove_vocab:
                        q.append(glove_vocab[word])
                    else:
                        q.append(0)
                tmp = [q, pr, nr]
                wq_train_data.append(tmp)
        np.save(save_train_data, wq_train_data)

        # lodaing webquestion testing data
        wq_test_data = []
        with open(wq_test_file) as rela:
            for line in rela:
                q, pr, nr = [], [], []
                line = line.split("	")
                posi_rela, nega_rela, question = line[0], line[1], line[2]
                # print "posi_rela", posi_rela, "nega_rela", nega_rela, "quesiton", question

                # turn negative relations into indexes
                nega_rela = nega_rela.split(" ")
                for nega in nega_rela:
                    if int(nega) >= len(wq_index2relations):
                        print"***index:", int(nega)
                    relation = wq_index2relations[int(nega)]
                    nr.append(relation)
                # turn positive relations into indexes
                posi_rela = posi_rela.split(" ")
                for posi in posi_rela:
                    relation = wq_index2relations[int(posi)]
                    pr.append(relation)
                # turn question into indexes
                question = question.split(" ")
                for word in question:
                    if word in glove_vocab:
                        q.append(glove_vocab[word])
                    else:
                        q.append(0)
                tmp = [q, pr, nr]
                wq_test_data.append(tmp)
        np.save(save_test_data, wq_test_data)
        print "***Load WebQuestion data done***"

    def generate_smaller_glove(self, glove_file, train_file, test_file, relations_file):
        glove_vocab, glove_vec = self.load_glove(glove_file)

        new_glove_vocab = {}
        new_glove_vec = torch.zeros(1, 300)
        new_index = 1
        with open(relations_file) as relations:
            for line in relations:
                tmp = line.replace("..", " ").replace("_", " ").replace(".", " ").replace("\n", "")
                tmp = tmp.split(" ")
                for word in tmp:
                    if word in glove_vocab and word not in new_glove_vocab:
                        new_glove_vocab[word] = new_index
                        new_index += 1
                        vec = glove_vec[glove_vocab[word]]
                        vec = torch.unsqueeze(vec, 0)
                        # print vec.size(), type(vec)
                        new_glove_vec = torch.cat((new_glove_vec, vec), 0)

        print "smaller glove vec: ", new_glove_vec.size(), type(new_glove_vec)
        print "smaller glove vocab: ", len(new_glove_vocab), type(new_glove_vocab)

        with open(train_file) as rela:
            for line in rela:
                line = line.split("	")
                if len(line) != 3:
                    print "error", line
                question = line[2]
                tmp = question.split(" ")
                for word in tmp:
                    if word in glove_vocab and word not in new_glove_vocab:
                        new_glove_vocab[word] = new_index
                        new_index += 1
                        vec = glove_vec[glove_vocab[word]]
                        vec = torch.unsqueeze(vec, 0)
                        # print vec.size(), type(vec)
                        new_glove_vec = torch.cat((new_glove_vec, vec), 0)
        print "new vocab and vec"
        print new_glove_vec.size(), type(vec)
        print len(new_glove_vocab), type(new_glove_vocab)

        with open(test_file) as rela:
            for line in rela:
                line = line.split("	")
                if len(line) != 3:
                    print "error", line
                question = line[2]
                tmp = question.split(" ")
                for word in tmp:
                    if word in glove_vocab and word not in new_glove_vocab:
                        new_glove_vocab[word] = new_index
                        new_index += 1
                        vec = glove_vec[glove_vocab[word]]
                        vec = torch.unsqueeze(vec, 0)
                        # print vec.size(), type(vec)
                        new_glove_vec = torch.cat((new_glove_vec, vec), 0)
        print "new vocab and vec"
        print new_glove_vec.size(), type(vec)
        print len(new_glove_vocab), type(new_glove_vocab)
        self.save_pickle(new_glove_vocab, "glove/smaller_glove_vocab.pickle")
        torch.save(new_glove_vec, "glove/smaller_glove_vec")
        print "--------smaller_glove_generation_complete--------------------------"


    def save_pickle(self, data, save_file):
        """
        save a dictionary into a pickle file
        :param data:
        :param save_file:
        :return:
        """
        print "Pickle saving", save_file
        with open(save_file, "wb") as sf:
            pickle.dump(data, sf)

    def load_pickle(self, read_file):
        """
        read a dictionary from a pickle file
        :param read_file:
        :return:
        """
        print "Pickle reading ", read_file
        with open(read_file, "rb") as rf:
            return pickle.load(rf)

    def load_glove(self, glove_file):
        """
        load glove model and add a all-zero row in the index 0 row, and save mapping word2index and matrix index2vector
        :param glove_file: glove txt file
        :return: none
        """
        print "Load original huge glove pre-trained vectors"
        vocab = {}
        vec = []
        index = 1
        with open(glove_file, "rb") as gl:
            for line in gl:
                tmp = []
                words = line.split(" ")
                for i in range(1, len(words)):
                    tmp.append(float(words[i]))
                vec.append(tmp)
                vocab[words[0]] = index
                index = index + 1
        vec = torch.FloatTensor(vec)
        vec = torch.cat((torch.zeros(1, 300), vec), 0)
        self.save_pickle(vocab, "glove/glove_vocab.pickle")
        torch.save(vec, "glove/glove_vec")
        return vocab, vec


# """test"""
# loader = loader()
# '''generate smaller glove vocabulary and vector '''
# # loader.generate_smaller_glove("glove/glove.42B.300d.txt",
# #                               "wq/wq_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt",
# #                               "wq/wq_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt",
# #                               "wq/wq_relations/relations.txt")
# '''load webquestion data'''
# smaller_glove_vocab = loader.load_pickle("glove/smaller_glove_vocab.pickle")
# loader.load_wq_relations(smaller_glove_vocab,
#                          "wq/wq_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt",
#                          "wq/wq_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt",
#                          "wq/wq_relations/relations.txt",
#                          "wq/wq_train_data",
#                          "wq/wq_test_data",
#                          "wq/wq_index2realtion")


