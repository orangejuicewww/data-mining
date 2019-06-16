import os

import numpy as np
import pandas as pd
from gensim.models import word2vec

from data_mining.config import Config


class DataHelper(object):
    def __init__(self):
        self.sequence_length = 500
        self.init()
        self.embedding_dim = self.model.wv.vector_size
        self.vocab_size = len(self.model.wv.vocab)
        self.num_tags = len(self.tag2id)

    def init(self):
        self.model = word2vec.Word2Vec.load(Config.w2v_model_path)
        tags = ['悲', '惧', '乐', '怒', '思', '喜', '忧']
        self.tag2id = {tag: index for index, tag in enumerate(tags)}
        self.id2tag = {tag_id: tag for tag, tag_id in self.tag2id.items()}
        # self.word2id = self.model.wv.vocab["云"].index
        # self.id2word = self.model.wv.index2word
        self.pre_embeddings = self.model.wv.vectors

    def pad(self, x):
        """ word_id 的列表，补零或切断到等长
        :param x: sentence id;[14,23,12];     ["我","爱","北京"]
        :return: x  self.sequence_length 长度的word_id 列表
        """
        if len(x) < self.sequence_length:
            x = x + [0] * (self.sequence_length - len(x))
        # Lists of the same level (the same numpy axis) need to be the same size. Otherwise you get an array of lists.
        return x[:self.sequence_length]

    out_vocab = set()

    def get_x_data(self, sentence):
        """ word转id并补齐
        :param sentence: ["我","爱","北京"]
        :return: [2,3,4,5,6,7,2,4,2, ...]；固定长度的word id 列表
        """
        index = self.model.wv.vocab["曰"].index
        x = []
        for _word in sentence:
            if _word in self.model:
                index = self.model.wv.vocab[_word].index
            else:
                self.out_vocab.add(_word)
            x.append(index)
        x = self.pad(x)  # pad=0
        return x

    def get_y_data(self, tags, y_vectorize=False):
        """
        :param tags: tag_names，句子对应的标签列表
        :param y_vectorize:  向量化
        :return: tag_ids 或者 num_classes长度的向量
        """
        y = [self.tag2id[tag] for tag in tags]
        if y_vectorize:
            y = [1 if tag_id in y else 0 for tag_id in range(self.num_tags)]
        return y

    def load_data(self, reuse=True):
        if os.path.exists(Config.train_data_path) and reuse:
            data_df = pd.read_csv(Config.train_data_path)
        else:
            tag_keys = {'悲': ['愁', '恸', '痛', '寡', '哀', '伤', '嗟'],
                        '惧': ['谗', '谤', '患', '罪', '诈', '惧', '诬'],
                        '乐': ['悦', '欣', '乐', '怡', '洽', '畅', '愉'],
                        '怒': ['怒', '雷', '吼', '霆', '霹', '猛', '轰'],
                        '思': ['思', '忆', '怀', '恨', '吟', '逢', '期'],
                        '喜': ['喜', '健', '倩', '贺', '好', '良', '善'],
                        '忧': ['恤', '忧', '痾', '虑', '艰', '遑', '厄']}
            # data_path = os.path.join(data_dir, "poems.csv")
            data_df = pd.read_csv(Config.poem_data_path)
            data_df.dropna(subset=['内容'], inplace=True)
            data_df["标签"] = data_df["内容"].apply(
                lambda poem: [tag for word in poem for tag, keys in tag_keys.items() if word in keys])
            data_df.to_csv(Config.train_data_path)
        data_df.dropna(subset=['内容'], inplace=True)
        return data_df["内容"].values, data_df["标签"].values

    def get_data(self, y_vectorize=False):
        """ keras models
        :param data_type:
        :return: 句子word转id后的列表；对应的tag_id列表
        """
        poems, tags = self.load_data()
        x_data = []
        y_data = []
        for _sentence, _tag in zip(poems, tags):
            if not _tag or not _sentence:
                continue
            try:
                tag = eval(_tag)
            except:
                # import ipdb
                # ipdb.set_trace()
                continue
            x = self.get_x_data(_sentence)  # pad=0,unk=1
            y = self.get_y_data(tag, y_vectorize=y_vectorize)
            x_data.append(x)
            y_data.append(y)
        print("not in vocab : {}".format(len(self.out_vocab)))
        print("not in vocab : {}".format(self.out_vocab))
        return np.asarray(x_data), np.asarray(y_data)
