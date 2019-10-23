# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2019-02-26 11:44:15
@Author: Ma-Dan
@Date: 2019-09-01 18:00:00
'''
import copy
import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids

class XLNetDataUtils(object):

    def __init__(self, sp_model, entry="train", batch_size=1):
        self.sp_model = sp_model

        self.data = []
        self.input_size = 0
        self.vocab = {}
        self.batch_data = []
        self.tag_map = {"O":0}

        self.batch_size = batch_size
        self.max_seq_length = 512

        if entry == "train":
            self.data_path = "data/train.txt"
        elif entry == "dev":
            self.data_path = "data/dev.txt"

        self.load_data()
        self.prepare_batch()

    def load_data(self):
        # load data
        # add vocab
        # covert to one-hot
        ntokens = []
        target = []
        train_nums = 0
        with open(self.data_path, encoding='UTF-8') as f:
            for line in f:
                line = line.rstrip()
                train_nums += 1
                try:
                    word, tag = line.split()
                except Exception as error:
                    word = "。"
                    tag = "O"
                    if line == "":
                        # process long sentence
                        if len(target) >= self.max_seq_length:
                            self.cut_long_sentence(ntokens, target)
                            ntokens = []
                            target = []
                            continue

                        self.add_one_sentence(ntokens, target)
                        ntokens = []
                        target = []
                    continue
                ntokens.append(word)
                target.append(tag)

    def cut_long_sentence(self, ntokens, target):
        if len(ntokens) < self.max_seq_length:
            self.add_one_sentence(ntokens, target)
            return

        for i in range(self.max_seq_length):
            if target[self.max_seq_length-i] == "O":
                break

        ntokens_left = ntokens.copy()[0:self.max_seq_length-i-1]
        target_left = target.copy()[0:self.max_seq_length-i-1]
        self.add_one_sentence(ntokens_left, target_left)

        ntokens_right = ntokens.copy()[self.max_seq_length-i:]
        target_right = target.copy()[self.max_seq_length-i:]
        self.cut_long_sentence(ntokens_right, target_right)

    def add_one_sentence(self, ntokens, target):
        tag_ids = self.convert_tag(target)
        text = ''.join(ntokens)
        text.replace('…', '.')
        text.replace('℃', 'C')
        text_preprocessed = preprocess_text(text)
        inputs_ids = encode_ids(self.sp_model, text_preprocessed)
        segment_ids = [0] * len(inputs_ids)
        input_mask = [1] * len(inputs_ids)

        data = [ntokens, tag_ids, inputs_ids, segment_ids, input_mask]
        self.data.append(data)

    def convert_tag(self, tag_list):
        tag_ids = []
        for tag in tag_list:
            if tag not in self.tag_map:
                self.tag_map[tag] = len(self.tag_map.keys())
            tag_ids.append(self.tag_map.get(tag, 0))
        return tag_ids

    def prepare_batch(self):
        '''
            prepare data for batch
        '''
        index = 0
        while True:
            if index+self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[2]) for i in c_data])
        padded_data = []
        for i in c_data:
            ntokens, tag_ids, inputs_ids, segment_ids, input_mask = i
            # ntokens = ntokens + (max_length - len(ntokens)) * ["**NULL**"]
            tag_ids = tag_ids + (max_length - len(tag_ids)) * [0]
            inputs_ids = inputs_ids + (max_length - len(inputs_ids)) * [0]
            segment_ids = segment_ids + (max_length - len(segment_ids)) * [0]
            input_mask = input_mask + (max_length - len(input_mask)) * [0]
            if (len(tag_ids) == len(inputs_ids) == len(segment_ids) == len(input_mask)):
                padded_data.append(
                    [ntokens, tag_ids, inputs_ids, segment_ids, input_mask]
                )
            else:
                #print("tokenized length error")
                print(ntokens, tag_ids, inputs_ids, segment_ids, input_mask)

        return padded_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data