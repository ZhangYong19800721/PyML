# -*- coding: utf-8 -*-
"""
    聊天机器人
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

def printLines(file, n=10):
    """打印文件的前n行"""
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

def loadLines(fileName, fields):
    """
        从文件fileName中读取数据，字符串列表fields指定了数据列的含义
    """
    # 对于movie_lines.txt 文件 fields = ["lineID", "characterID", "movieID", "character", "text"]
    lines = {} # 从行ID到行对象的列表
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f: # 读取一行
            values = line.split(" +++$+++ ") # 数据是被字符串" +++$+++ "分隔的
            lineObj = {} # 这是一个行对象
            for i, field in enumerate(fields):
                lineObj[field] = values[i] # 将第i个数据填入field对应的位置
            lines[lineObj['lineID']] = lineObj # 建立从行ID到行对象的映射
    return lines

def loadConversations(fileName, lines, fields):
    """
        加载对话数据。Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
        fileName 对话数据文件，movie_conversations.txt
        lines    字典，从行ID到行对象的映射
        fields   数据含义列表
    """
    # 对于 movie_conversations.txt fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ") # 数据是被字符串" +++$+++ "分隔的
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # 利用eval函数直接将字符串转换为列表
            lineIds = eval(convObj["utteranceIDs"])
            # 重新组装对话行
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# 从对话中抽取语句对
# 输入参数 conversations 是一个包含对话对象的列表
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations: # 对所有的对话进行循环
        # 对对话中所有的行进行循环
        for i in range(len(conversation["lines"]) - 1):  # 忽略最后一行，因为最后一行没有对应的答句
            inputLine = conversation["lines"][i]["text"].strip() # 对话的第i个句子作为输入
            targetLine = conversation["lines"][i+1]["text"].strip() # 对话的第i+1个句子作为输出
            # 过滤掉输入行或者输出行为空的情况
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

PAD_token = 0  # 用来填充较短的句子
SOS_token = 1  # 句子开始的标记
EOS_token = 2  # 句子结束的标记

class Voc:
    """
            词汇表
    功能：
        1.将词映射为一个索引值
        2.将索引值映射为词
        3.对每个词进行计数
        4.对总次数进行计数
        5.接口函数addWord可以将某个词加到词汇表中
        6.接口函数addSentence可以将某个句子中的词加入到词汇表中
        7.接口函数trim去掉不常用的词
    """
    def __init__(self, name):
        self.name = name # 词汇表的名字
        self.trimmed = False # 是否已经剪切掉不常用的词汇
        self.word2index = {} # 从词到索引的映射表
        self.word2count = {} # 词的计数
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"} # 从索引到词的映射表
        self.num_words = 3  # 词汇量计数

    def addSentence(self, sentence):
        # 输入参数 sentence 是一个字符串句子，词中间是空格隔开
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index: # 如果该词不在词表中
            self.word2index[word] = self.num_words # 建立词到索引的映射
            self.word2count[word] = 1 # 初始化词计数为1
            self.index2word[self.num_words] = word # 建立从索引到词映射
            self.num_words += 1 # 词总数加1
        else: # 如果该词已经在词表中，则词计数加1
            self.word2count[word] += 1

    # 去除不常用词
    # 输入参数：min_count,当词计数小于min_count时，即为不常用词
    def trim(self, min_count):
        if self.trimmed: # 如果该词表已经被剪切过了，就直接返回
            return
        self.trimmed = True # 先将剪切标志设置为True

        keep_words = [] # 被保留的词列表

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        # 重新初始化词表
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        # 将被保留词列表中的词加入到词汇表中
        for word in keep_words:
            self.addWord(word)

MAX_LENGTH = 10  # 考虑的最大句子长度为10

# 将Unicode字符串转换为普通的ASCII, 参考
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 将字符串转换为小写，剪除左右的空格，去除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# 读取问答数据对，并返回一个词表对象
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# 如果问句和答句的长度都小于MAX_LENGTH，则返回True
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

MIN_COUNT = 3    # 最小词计数，少于这个计数的词被认为是非常用词
def trimRareWords(voc, pairs, MIN_COUNT):
    # 从词表中剪除非常用词
    voc.trim(MIN_COUNT)
    # 剪除包含非常用词的句子对
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

# 将句子转换为词索引列表
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

if __name__ == "__main__":
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    printLines(os.path.join(corpus, "movie_lines.txt"))

    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    printLines(datafile)

    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    pairs = trimRareWords(voc, pairs, MIN_COUNT)

    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)