from functools import reduce as rd
import pandas as pd
import os
from itertools import product as prod
from functools import partial as pf
import time
from neuralNet import Neural_Net
from copy import copy
import random
from Population import Population

class CrossValidation:
    def __init__(self, data):
        self.data = data
        self.NN = Neural_Net(data)
        self.Pop = Population(data)

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


    def error_from_df(self, model, dataset, hyp_df):
        error_func = self.error_from_series(model, dataset)
        def f(i):
            hyps = hyp_df.loc[i, :]
            return error_func(hyps)
        return f

    def error_from_series(self, model, dataset):
        if model == 'GA': f = self.error_GA(dataset)
        '''
        if model == 'DE': f = self.error_DE(dataset)
        if model == 'PSO': f = self.error_PSO(dataset)
        '''
        return f

    def error_GA(self, dataset):
        def f(hyps):
            chr = self.Pop.run(num_hidden=hyps["num_hidden"], p_c=hyps["p_c"], p_m=hyps["p_m"], pop_size=hyps["pop_size"])
            return self.Pop.predict_with_chr(dataset, chr, hyps["num_hidden"], error=True)
        return f

    def tuneHyps(self, model, train_dict, fold, hyp_list, start_hyp_dict):
        def tuneHyp(hyp_dict, fix, find):
            hyp_spaces = tuple([[random.choice(hyp_dict[hyp])] if hyp in fix else hyp_dict[hyp] for hyp in hyp_list])
            my_space = pd.Series(prod(*hyp_spaces))
            df_size = len(my_space)
            cols = list(zip(*my_space))
            col_titles = hyp_list
            data = zip(col_titles, cols)
            error_df = pd.DataFrame(index=range(len(my_space)))
            for (title, col) in data:
                error_df[title] = col
            error = self.error_from_df(model = model, dataset = train_dict[fold], hyp_df = error_df)
            error_column = pd.Series(range(df_size)).map(error).values
            min_error = error_column.min()
            error_df["Error"] = error_column
            error_df.to_csv(os.getcwd() + '\\' + str(self.data) + '\\' +
                            "{}_Tune_{}_Fix_{}_Fold_{}.csv".format(str(self.data), find, fix, fold))
            best_row = error_df.loc[lambda df: df["Error"] == min_error].iloc[0]
            return best_row[find]

        def linearSearch(hyp_dict, toSearch):
            while len(toSearch) > 0:
                hypToFind = toSearch.pop()
                bestValue = tuneHyp(hyp_dict, toSearch, hypToFind)
                hyp_dict[hypToFind] = [bestValue]
            return pd.Series(hyp_list, index=hyp_list).map(lambda hyp: hyp_dict[hyp][0])

        return linearSearch(start_hyp_dict, set(hyp_list))









