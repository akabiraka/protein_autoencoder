import os
import numpy as np
import math

from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import constants as CONSTANTS


class FastaDataset(Dataset):
    """docstring for FastaDataset."""

    def __init__(self, filename, encoding_mode='numeric', pad_mode=None):
        """
        encoding_mode = "numeric", 'one_hot'
        pad_mode = None, 'front', 'end'
        """
        super(FastaDataset, self).__init__()vg
        file = open(filename)
        self.records = list(enumerate(parse(file, CONSTANTS.FASTA)))
        self.encoding_mode = encoding_mode
        self.AALetters = CONSTANTS.AMINO_ACID_22
        self.AA_c2i_dict, self.AA_i2c_dict = self.__get_amino_acid_seq_dict()
        self.n_letters = len(self.AALetters)
        self.max_len, self.min_len, self.avg_len = self.get_max_length()
        self.pad_mode = pad_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        id, record = self.records[i]
        embedding = 0.0
        if self.encoding_mode == 'numeric':
            embedding = self.seq_to_numeric(record.seq, self.pad_mode)
        elif self.encoding_mode == 'one_hot':
            embedding = self.seq_2_tensor(record.seq, self.pad_mode)
        return embedding

    def ger_record(self, i):
        id, record = self.records[i]
        return record

    def __get_amino_acid_seq_dict(self):
        char2Int = {}
        int2Char = {}
        for i, char in enumerate(self.AALetters):
            char2Int[char] = i + 1
            int2Char[i + 1] = char
        return char2Int, int2Char

    def get_max_length(self):
        lengths = []
        for id, record in self.records:
            lengths.append(len(record.seq))
        return max(lengths), min(lengths), math.floor(sum(lengths) / len(lengths))

    def pad_seq(self, embedding, value=0, mode='front'):
        """
            mode='None', 'front', 'end'
        """
        pad_len = self.max_len - len(embedding)
        if mode == None:
            pass
        elif mode == 'front':
            embedding = np.pad(embedding, (pad_len, 0),
                               'constant', constant_values=(value, value))
        elif mode == 'end':
            embedding = np.pad(embedding, (0, pad_len),
                               'constant', constant_values=(value, value))

        return embedding

    def seq_to_numeric(self, seq, pad_mode=None):
        numeric = []
        for i, letter in enumerate(seq):
            if letter not in self.AA_c2i_dict:
                print("'{}' does not exist in AMINO_ACID_letters".format(letter))
                return
            numeric.append(self.AA_c2i_dict.get(letter))
        embedding = self.pad_seq(numeric, mode=pad_mode)
        return torch.from_numpy(np.array(embedding))

    def letter_2_tensor(self, letter):
        embedding = torch.zeros(1, self.n_letters)
        index = self.AA_c2i_dict.get(letter)
        embedding[0][index] = 1
        return embedding

    def seq_2_tensor2(self, seq):
        embedding = torch.zeros(len(seq), 1, self.n_letters)
        for seq_index, letter in enumerate(seq):
            if letter not in self.AA_c2i_dict:
                print("'{}' does not exist in AMINO_ACID_letters".format(letter))
                return
            letter_index = self.AA_c2i_dict.get(letter)
            embedding[seq_index][0][letter_index] = 1

        return embedding

    def seq_2_tensor(self, seq, pad_mode=None):
        numeric = self.seq_to_numeric(seq, pad_mode)
        one_hot = F.one_hot(numeric, num_classes=self.n_letters)
        return one_hot
