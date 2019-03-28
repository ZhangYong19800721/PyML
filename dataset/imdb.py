# -*- coding: utf-8 -*-
"""
    IMDB电影评论数据集
"""

import six.moves.cPickle as pk

# 因为imdb.dict.pkl文件是使用python2写入的，而python3和python2在字符串方面不兼容，所以需
# 要这个类进行数据转换，以方便读入数据
class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

# IMDB数据集的词汇表
class IMDB_DICT(object):
    def __init__(self):
        dict_pkl = open("../data/imdb.dict.pkl","r")
        self.word2idx = pk.load(StrToBytes(dict_pkl),encoding='UTF-8')
        dict_pkl.close()
        self.idx2word = {}
        for k,v in self.word2idx.items():
            self.idx2word[v] = k
        

def remove_unk(x):
    return [[1 if w >= 100000 else w for w in sen] for sen in x]

def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))

class train_set(object):
    def __init__(self):
        imdb_pkl = open("../data/imdb.pkl","rb")
        train_data = pk.load(imdb_pkl)
        test_data = pk.load(imdb_pkl)
        imdb_pkl.close()        
        self.datas = train_data[0] # 训练数据
        self.label = train_data[1] # 训练标签
        # 下面进行一些数据的预处理
        # 第一步，截断，句子的长度不超过max_sentence_length        
        max_sentence_length = 100 # 最大句子长度
        new_datas = []
        new_label = []
        for x,y in zip(self.datas,self.label):
            if(len(x)<=max_sentence_length):
                new_datas.append(x)
                new_label.append(y)
        self.datas = new_datas
        self.label = new_label
        del(new_datas)
        del(new_label)
        
        # 第二步，去除不常用的单词
        self.datas = remove_unk(self.datas)
        
        # 第三步，将句子按照长短进行排序
        sorted_idx = len_argsort(self.datas)
        self.datas = [self.datas[i] for i in sorted_idx]
        self.label = [self.label[i] for i in sorted_idx]
        
        # 初始化minibatch的大小为1
        self.minibatch_size = 1

    def __len__(self):
        return len(self.datas) // self.minibatch_size

    def __getitem__(self, item):
        if self.minibatch_size > 1:
            return self.datas[(item * self.minibatch_size):((item + 1) * self.minibatch_size), :], \
                   self.label[(item * self.minibatch_size):((item + 1) * self.minibatch_size), :]
        else:
            return self.datas[item], self.label[item]

    def set_minibatch_size(self, size):
        self.minibatch_size = size


class test_set(object):
    def __init__(self):
        imdb_pkl = open("../data/imdb.pkl","rb")
        train_data = pk.load(imdb_pkl)
        test_data = pk.load(imdb_pkl)
        imdb_pkl.close()        
        self.datas = test_data[0] # 测试数据
        self.label = test_data[1] # 测试标签
        # 下面进行一些数据的预处理
        # 第一步，截断，句子的长度不超过max_sentence_length        
        max_sentence_length = 100 # 最大句子长度
        new_datas = []
        new_label = []
        for x,y in zip(self.datas,self.label):
            if(len(x)<=max_sentence_length):
                new_datas.append(x)
                new_label.append(y)
        self.datas = new_datas
        self.label = new_label
        del(new_datas)
        del(new_label)
        
        # 第二步，去除不常用的单词
        self.datas = remove_unk(self.datas)
        
        # 第三步，将句子按照长短进行排序
        sorted_idx = len_argsort(self.datas)
        self.datas = [self.datas[i] for i in sorted_idx]
        self.label = [self.label[i] for i in sorted_idx]
        
        # 初始化minibatch的大小为1
        self.minibatch_size = 1

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, item):
        if self.minibatch_size > 1:
            return self.datas[(item * self.minibatch_size):((item + 1) * self.minibatch_size), :], \
                   self.label[(item * self.minibatch_size):((item + 1) * self.minibatch_size)]
        else:
            return self.datas[item], self.label[item]

    def set_minibatch_size(self, size):
        self.minibatch_size = size


if __name__ == '__main__':
    imdb_dict = IMDB_DICT()
    train_data = train_set()
    review = [imdb_dict.idx2word[x] for x in train_data.datas[0]]
    print(' '.join(review))
    print(train_data.label[0])
    
    test_data = test_set()
    review = [imdb_dict.idx2word[x] for x in test_data.datas[0]]
    print(' '.join(review))
    print(test_data.label[0])