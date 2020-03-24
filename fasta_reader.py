import os
import numpy as np
import math

from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from constants import *


class FastaReader(object):
    """docstring for FastaReader."""

    def __init__(self, filename):
        super(FastaReader, self).__init__()

        extension = os.path.splitext(filename)[1][1:]
        does_exist = os.path.isfile(filename)

        if does_exist and extension == FASTA:
            self.filename = filename
        else:
            print("Incorrect fasta file or file not exist")

    def __get_records(self):
        file = open(self.filename)
        records = parse(file, FASTA)
        return records

    def get_raw_records(self):
        return self.__get_records()

    def __amino_acid_seq_dictionary(self):
        char2Int = {}
        int2Char = {}
        for i, char in enumerate(AMINO_ACID_22):
            char2Int[char] = i + 1
            int2Char[i + 1] = char
        return char2Int, int2Char

    def __get_max_length(self):
        records = self.__get_records()
        lengths = []
        for record in records:
            lengths.append(len(record.seq))

        return max(lengths), min(lengths), math.floor(sum(lengths) / len(lengths))

    def seq_2_numeric_encoding(self, seq):
        numeric_seq = []
        amino_acid_c2i_dict, _ = self.__amino_acid_seq_dictionary()
        for i, v in enumerate(seq):
            if v in amino_acid_c2i_dict:
                numeric_seq.append(amino_acid_c2i_dict.get(v))
            else:
                print("'{}' does not exist in AMINO_ACID_20".format(v))

        return np.array(numeric_seq)

    def pad_seq(self, numeric_seq, max_len, value=0, mode='front'):
        """
            mode='None', 'front', 'end'
        """
        pad_len = max_len - len(numeric_seq)
        if mode == None:
            pass
        elif mode == 'front':
            numeric_seq = np.pad(numeric_seq, (pad_len, 0),
                                 'constant', constant_values=(value, value))
        elif mode == 'end':
            numeric_seq = np.pad(numeric_seq, (0, pad_len),
                                 'constant', constant_values=(value, value))

        return numeric_seq

    def fasta_2_numeric_encoding(self, pad_mode=None):
        """
            pad_mode='None', 'front', 'end'
        """
        all_numeric_seq = []
        max_length, _, _ = self.__get_max_length()
        records = self.__get_records()
        for i, record in enumerate(records):
            numeric_seq = self.seq_2_numeric_encoding(record.seq)
            numeric_seq = self.pad_seq(numeric_seq, max_length, mode=pad_mode)
            print("{} th sequence's shape is {}".format(
                i + 1, numeric_seq.shape))
            all_numeric_seq.append(numeric_seq)

        return np.array(all_numeric_seq)
