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
                (q, r) = (n // k, n % k)  #q is the quotient; r is the remainder
                j = 0
                for i in range(k):
                    z = (i + c) % k       #start in the appropriate bucket based off previous remainder
                    p[z] = p[z] + [classdf.at[x, 'index'] for x in range(j, j + q + int(i < r))]
                    j += q + int(i < r)
                return (p, c + r)
            c = 0
            for cl in self.data.classes:
                classdf = df[df['Target'] == cl].reset_index()          #take data from one class
                (p, c) = class_partition(classdf, p, c)                 #partition that data of that one class
        else:
            sorted_df = df.sort_values(by=['Target']).reset_index()
            n = sorted_df.shape[0]
            (q, r) = (n // k, n % k)
            for i in range(k):
                p[i] = p[i] + [sorted_df.at[i + c * k, 'index'] for c in range(q + int(i < r))] #regression partition
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
            train_dict[i] = df.filter(items=train_index, axis=0)    #define the training set at fold i
            test_dict[i] = df.filter(items=partition[i], axis=0)    #define the test set at fold i
        return (train_dict, test_dict)

    '''
    @param model: a string that tells the model we are using (either, GA, DE, or PSO)
    @param num_hidden: an integer that tells us how many hidden layers we are using
    @param dataset: a pandas dataframe that is a subset of the data
    @param hyp_df: a dataframe where the columns are the hyperparameters
    @return: function that inputs an index of the dataframe and returns error based on the rows hyperparameters
    '''
    def error_from_df(self, model, num_hidden, dataset, hyp_df):
        error_func = self.error_from_series(model, num_hidden, dataset)  #an error function that takes index as input
        def f(i):
            hyps = hyp_df.loc[i, :]
            return error_func(hyps)
        return f
    '''
    @param model: a string that tells the model we are using (either, GA, DE, or PSO)
    @param num_hidden: an integer that tells us how many hidden layers we are using
    @param dataset: a pandas dataframe that is a subset of the data
    @return: function that inputs a series of hyperparameters and returns the error of given model for those hyps
    '''
    def error_from_series(self, model, num_hidden, dataset):
        if model == 'GA': f = self.error_GA(num_hidden, dataset)
        if model == 'DE': f = self.error_DE(num_hidden, dataset)
        if model == 'PSO': f = self.error_PSO(num_hidden, dataset)
        return f
    '''
    @param num_hidden: an integer that tells us how many hidden layers we are using
    @param dataset: a pandas dataframe that is a subset of the data
    @return: function that inputs a series of hyperparameters and returns the error of GE for those hyps
    '''
    def error_GA(self, num_hidden, dataset):
        def f(hyps):
            print("Find Best Chromosome!")
            chr_time = time.time()
            chr = self.Pop.run(num_hidden=num_hidden, dataset = dataset, p_c=hyps["p_c"], p_m=hyps["p_m"], pop_size=hyps["pop_size"])
            print("Time to find best chromosome: {} Seconds.".format(time.time() - chr_time))
            return self.Pop.predict_with_chr(dataset, chr, num_hidden, error=True)
        return f

    def error_DE(self, num_hidden, dataset):
        def f(hyps):
            chr = self.Pop.runDE(num_hidden=num_hidden, dataset = dataset, p_c=hyps["p_c"], p_m=hyps["p_m"], pop_size=hyps["pop_size"])
            return self.Pop.predict_with_chr(dataset, chr, num_hidden, error=True)
        return f
    
    def error_PSO(self, num_hidden, dataset):
        def f(hyps):
            chr = self.Pop.runPSO(num_hidden=num_hidden, dataset = dataset, w=hyps["p_w"], c=hyps["p_c"], pop_size=hyps["pop_size"])
            return self.Pop.predict_with_chr(dataset, chr, num_hidden, error=True)
        return f

    '''
    @param model: a string that tells the model we are using (either, GA, DE, or PSO)
    @param num_hidden: an integer that tells us how many hidden layers we are using
    @param train_dict: a dictionary that inputs the fold and returns a training set
    @param fold: an integer that represents a fold for k-fold cross validation
    @param hyp_list: a list of hyperparameters we are tuning for
    @param start_hyp_dict: a starting dictionary that inputs hyperparameter and returns list of values to tune over
    @return: a dictionary that inputs hyperparameters and returns the best value for that hyperparameter
    '''
    def tuneHyps(self, model, num_hidden, train_dict, fold, hyp_list, start_hyp_dict):
        '''
        @param hyp_dict: the current hyperparameter dictionary we are tuning over
        @param fix: a set of hyperparameters that we want fixed at one value for tuning
        @param find: the hyperparameter we are tuning for to find the best value for
        @return: the best value for the given hyperparameter among the options
        '''
        def tuneHyp(hyp_dict, fix, find):
            hyp_spaces = tuple([[random.choice(hyp_dict[hyp])] if hyp in fix else hyp_dict[hyp] for hyp in hyp_list])
            my_space = pd.Series(prod(*hyp_spaces))  #a cartesian product where only one hyperparameter varies
            df_size = len(my_space)
            cols = list(zip(*my_space))          #columns are values of hyperparameters for their respective hyperparameter
            col_titles = hyp_list                #the hyperparameter list gives the name of the columns
            data = zip(col_titles, cols)
            error_df = pd.DataFrame(index=range(len(my_space)))
            for (title, col) in data:
                error_df[title] = col
            error = self.error_from_df(model, num_hidden, train_dict[fold], error_df)  #create error function
            print("Calcuating Error For Each Hyp Series")
            error_time = time.time()
            error_column = pd.Series(range(df_size)).map(error).values     #evaluate errors for hyperparameter df
            print("Time to compute errors: {} Seconds".format(time.time() - error_time))
            min_error = error_column.min()
            error_df["Error"] = error_column
            best_row = error_df.loc[lambda df: df["Error"] == min_error].iloc[0]  #find the row with the lowest error
            print("The best value for {} is {}.".format(find, best_row[find]))
            return (best_row[find], error_df)
        '''
        @param hyp_dict: the dictionary of hyperparameters and the values to tune over
        @param: the set of hyperparameters that still need to be tuned for
        @return: a dictionary that inputs a hyperparameter and returns the best value for that hyperparameter
        '''
        def linearSearch(hyp_dict, toSearch):
            error_dfs = []
            while len(toSearch) > 0:
                hypToFind = toSearch.pop()
                print("Tune for {}.".format(hypToFind))
                tune_time = time.time()
                (bestValue, error_df) = tuneHyp(hyp_dict, toSearch, hypToFind)
                print("Time to tune {}: {} Seconds".format(hypToFind, time.time() - tune_time))
                hyp_dict[hypToFind] = [bestValue]
                error_dfs.append(error_df)
            results_df = pd.concat(error_dfs).reset_index(drop=True)
            results_df.to_csv(os.getcwd() + '\\' + str(self.data) + '\\' +
                              "{}_{}_HiddenLayers_Error_Fold_{}.csv".format(str(self.data), num_hidden, fold))
            return pd.Series(hyp_list, index=hyp_list).map(lambda hyp: hyp_dict[hyp][0])

        return linearSearch(copy(start_hyp_dict), set(hyp_list))

    def analysis(self, model, num_hidden, hyp_list, start_hyp_dict):
        p = self.stratified_partition(10)
        (train_dict, test_dict) = self.training_test_dicts(self.data.df, p)
        analysisDF = pd.DataFrame(index = range(10), columns=hyp_list)
        error_column = []
        for fold in analysisDF.index:
            print("====================================================================================================")
            print("Training for Fold {}".format(fold))
            train_time = time.time()
            hyp_dict = self.tuneHyps(model, num_hidden, train_dict, fold, hyp_list, start_hyp_dict)
            print("Time to tune hyps for fold {}: {} Seconds".format(fold, time.time() - train_time))
            analysisDF.loc[fold, :] = pd.Series(hyp_dict)
            print("Test for Fold {}".format(fold))
            test_time = time.time()
            error_column.append(self.error_from_df(model, num_hidden, test_dict[fold], analysisDF)(fold))
            print("Time to test for fold {}: {} Seconds".format(fold, time.time() - test_time))
        analysisDF["Error"] = error_column
        analysisDF.to_csv(os.getcwd() + '\\' + str(self.data) + '\\' +
                          "{}_{}_HiddenLayers_Analysis.csv".format(str(self.data), num_hidden))


    def test(self, start_hyp_dict):
        p = self.stratified_partition(10)
        (train_dict, test_dict) = self.training_test_dicts(self.data.df, p)
        self.tuneHyps("GA", 0, train_dict, 0, ['p_c', 'p_m', 'pop_size'], start_hyp_dict)








