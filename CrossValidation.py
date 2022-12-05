from functools import reduce as rd
import pandas as pd
import os
from itertools import product as prod
from functools import partial as pf
import time
from neuralNet import Neural_Net
from copy import copy
import random

class CrossValidation:
    def __init__(self, data):
        self.data = data
        self.NN = Neural_Net(data)

    '''
    @param k: the number of folds
    @param: df: an optional subdataframe
    @return: a list of lists that represents a partition of the data's index
    '''
    def stratified_partition(self, k, df = None):
        if df is None: df = self.data.df
        p = [[] for i in range(k)]
        if self.data.classification:
            def class_partition(classdf, p, c):
                n = classdf.shape[0]
                (q, r) = (n // k, n % k)
                j = 0
                for i in range(k):
                    z = (i + c) % k
                    p[z] = p[z] + [classdf.at[x, 'index'] for x in range(j, j + q + int(i < r))]
                    j += q + int(i < r)
                return (p, c + r)
            c = 0
            for cl in self.data.classes:
                classdf = df[df['Target'] == cl].reset_index()
                (p, c) = class_partition(classdf, p, c)
        else:
            sorted_df = df.sort_values(by=['Target']).reset_index()
            n = sorted_df.shape[0]
            (q, r) = (n // k, n % k)
            for i in range(k):
                p[i] = p[i] + [sorted_df.at[i + c * k, 'index'] for c in range(q + int(i < r))]
        return p

    '''
    @param df: the dataframe
    @param partition: the partition we are using
    @return: a tuple a dictionary of the training sets and a dictionary of the test sets
    '''
    def training_test_dicts(self, df, partition=None):
        if partition is None: partition = self.stratified_partition(10)
        train_dict = {}
        test_dict = {}
        for i in range(len(partition)):
            train_index = rd(lambda l1, l2: l1 + l2, partition[:i] + partition[i + 1:])
            train_dict[i] = df.filter(items=train_index, axis=0)
            test_dict[i] = df.filter(items=partition[i], axis=0)
        return (train_dict, test_dict)

    def errorSearch(self, hyp_dict, fix, find, train_set):
        hyp_param_space = []
        for hyp in fix:
            hyp_param_space.append([random.choice(hyp_dict[hyp])])
