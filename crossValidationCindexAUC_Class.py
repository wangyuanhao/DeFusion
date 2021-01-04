#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import combinations, chain
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import os
rpy2.robjects.numpy2ri.activate()


def kFlodTestn(k, n, X, y, stratisfied=False):
    allIdx = np.arange(X.shape[0])
    testIdx = []

    if stratisfied:
        fold = StratifiedKFold(n_splits=k)
    else:
        fold = KFold(n_splits=k)

    # k fold split
    for train, test in fold.split(X, y):
        testIdx.append(test)

    # choose n test sets to make a larger test set
    for comb in list(combinations(testIdx, n)):
        unionTestIdx = np.array(list(chain.from_iterable(comb)))

        # return the unique values in allIdx that are not in unionTestIdx
        unionTrainIdx = np.setdiff1d(allIdx, unionTestIdx)
        yield unionTrainIdx, unionTestIdx


class crossValidationCindexAUC:
    # note: event before time
    def __init__(self, data, kFold, y_in_group, file_name, rand_seed):
        self.data = data
        self.kFold = kFold
        self.y_in_group = y_in_group
        self.file_name = file_name
        self.rand_seed = rand_seed # default 0

    def _data_split(self):

        kFold = self.kFold
        data = self.data
        X = data.values
        rand_seed = self.rand_seed

        # if self.y_in_group:
        #
        #     if "group_label" not in data.columns.tolist():
        #         print("'group_label' not found in data column!")
        #     y = np.reshape(data.loc[:, "group_label"].values, (-1, 1))
        #
        # else:
        y = np.reshape(data.loc[:, "follow_up_event"].values, (-1, 1))

        skf = StratifiedKFold(n_splits=kFold, shuffle=True, random_state=rand_seed)

        return skf.split(X, y), X, y


    def valid_coxCindex(self):

        file_name = self.file_name
        rand_seed = self.rand_seed

        if self.y_in_group:
            ro.r["source"]("computeCindex.R")

        else:
            ro.r["source"]("computeCindexinLatent.R")

        res_cox = {"trainCI": [], "trainCI.S.D.": [],
                   "testCI": [], "testCI.S.D": []}

        # for train, test in kFlodTestn(kFold, combineNFold, X, y, stratisfied=True):
        data_split, X, y = self._data_split()


        for train, test in data_split:
            Xtrain, Xtest, ytrain, ytest = X[train, :], X[test, :], y[train, :], y[test, :]

            # training_data = np.column_stack((Xtrain, ytrain))
            training_data = Xtrain
            # testing_data = np.column_stack((Xtest, ytest))
            testing_data = Xtest

            print("Xtrain: ", training_data.shape)
            print("Xtest: ", testing_data.shape)


            # transform numpy array to R object
            training_data_R = ro.r.matrix(training_data, nrow=training_data.shape[0],
                                          ncol=training_data.shape[1])

            testing_data_R = ro.r.matrix(testing_data, nrow=testing_data.shape[0],
                                         ncol=testing_data.shape[1])

            try:
                cindexRes = ro.r.computeCindex(training_data_R, testing_data_R)

                train_Cindex = cindexRes[0]
                res_cox["trainCI"].append(train_Cindex)

                train_Cindex_se = cindexRes[1]
                res_cox["trainCI.S.D."].append(train_Cindex_se)

                test_Cindex = cindexRes[2]
                res_cox["testCI"].append(test_Cindex)

                test_Cindex_se = cindexRes[3]
                res_cox["testCI.S.D"].append(test_Cindex_se)
            except:
                res_cox["trainCI"].append(np.nan)
                res_cox["trainCI.S.D."].append(np.nan)
                res_cox["testCI"].append(np.nan)
                res_cox["testCI.S.D"].append(np.nan)

        res_cox_df = pd.DataFrame.from_dict(res_cox)
        foutstring_cox = "_Cindex_cox_rand_seed=%d.csv" % rand_seed
        fout_cox = file_name + foutstring_cox
        # res_cox_df.to_csv(fout_cox, index=False)

        # change for 10CV
        return list(res_cox_df["trainCI"]), list(res_cox_df["testCI"])

    def valid_coxAUC(self):

        file_name = self.file_name
        rand_seed = self.rand_seed
        if self.y_in_group:
            ro.r["source"]("computeAUC.R")
        else:
            ro.r["source"]("computeAUCinLatent.R")

        res_AUC = {"train_auc_3y": [], "train_auc_3y_se": [],
                   "test_auc_3y": [], "test_auc_3y_se": []}

        # for train, test in kFlodTestn(kFold, combineNFold, X, y, stratisfied=True):
        data_split, X, y = self._data_split()
        for train, test in data_split:
            Xtrain, Xtest, ytrain, ytest = X[train, :], X[test, :], y[train, :], y[test, :]

            # training_data = np.column_stack((Xtrain, ytrain))
            training_data = Xtrain
            # testing_data = np.column_stack((Xtest, ytest))
            testing_data = Xtest

            print("Xtrain: ", training_data.shape)
            print("Xtest: ", testing_data.shape)

            # transform numpy array to R object
            training_data_R = ro.r.matrix(training_data, nrow=training_data.shape[0],
                                          ncol=training_data.shape[1])

            testing_data_R = ro.r.matrix(testing_data, nrow=testing_data.shape[0],
                                         ncol=testing_data.shape[1])

            try:
                aucRes = ro.r.computeAUC(training_data_R, testing_data_R)

                train_auc_3y = aucRes[0]
                res_AUC["train_auc_3y"].append(train_auc_3y)

                train_auc_3y_se = aucRes[1]
                res_AUC["train_auc_3y_se"].append(train_auc_3y_se)

                test_auc_3y = aucRes[2]
                res_AUC["test_auc_3y"].append(test_auc_3y)

                test_auc_3y_se = aucRes[3]
                res_AUC["test_auc_3y_se"].append(test_auc_3y_se)
            except:
                res_AUC["train_auc_3y"].append(np.nan)
                res_AUC["train_auc_3y_se"].append(np.nan)
                res_AUC["test_auc_3y"].append(np.nan)
                res_AUC["test_auc_3y_se"].append(np.nan)

        res_AUC_df = pd.DataFrame.from_dict(res_AUC)
        foutstring = "_AUC_rand_seed=%d.csv" % rand_seed
        fout = file_name + foutstring
        res_AUC_df.to_csv(fout, index=False)

        mean_res_AUC_df = res_AUC_df.loc[:, ["train_auc_3y", "test_auc_3y"]].mean(axis=0).T
        std_res_AUC_df = res_AUC_df.loc[:, ["train_auc_3y", "test_auc_3y"]].std(axis=0).T

        return mean_res_AUC_df.tolist() + std_res_AUC_df.tolist()

    def valid_svmCindex(self):
        file_name = self.file_name

        res_svm = {"trainCI": [], "trainCI.S.D.": [],
                   "testCI": [], "testCI.S.D.": []}

        if self.y_in_group:
            ro.r["source"]("computeSVM.R")
        else:
            ro.r["source"]("computeCindexinLatent.R")

        data_split, X, y = self._data_split()
        for train, test in data_split:
            Xtrain, Xtest, ytrain, ytest = X[train, :], X[test, :], y[train, :], y[test, :]

            # training_data = np.column_stack((Xtrain, ytrain))
            training_data = Xtrain
            # testing_data = np.column_stack((Xtest, ytest))
            testing_data = Xtest

            print("Xtrain: ", training_data.shape)
            print("Xtest: ", testing_data.shape)

            # transform numpy array to R object
            training_data_R = ro.r.matrix(training_data, nrow=training_data.shape[0],
                                          ncol=training_data.shape[1])

            testing_data_R = ro.r.matrix(testing_data, nrow=testing_data.shape[0],
                                         ncol=testing_data.shape[1])
            try:
                svmRes = ro.r.computeSVM(training_data_R, testing_data_R)
                res_svm["trainCI"].append(svmRes[0])
                res_svm["trainCI.S.D."].append(svmRes[1])
                res_svm["testCI"].append(svmRes[2])
                res_svm["testCI.S.D."].append(svmRes[3])
            except:
                print("Error in svm!")
                res_svm["trainCI"].append(np.nan)
                res_svm["trainCI.S.D."].append(np.nan)
                res_svm["testCI"].append(np.nan)
                res_svm["testCI.S.D."].append(np.nan)

        res_svm_df = pd.DataFrame.from_dict(res_svm)
        foutstring_svm = "_Cindex_svm.csv"
        fout_svm = file_name + foutstring_svm
        res_svm_df.to_csv(fout_svm, index=False)
        mean_res_svm_df = res_svm_df.loc[:, ["trainCI", "testCI"]].mean(axis=0).T
        std_res_svm_df = res_svm_df.loc[:, ["trainCI", "testCI"]].std(axis=0).T

        return mean_res_svm_df.tolist() + std_res_svm_df.tolist()

    def valid_rfCindex(self):

        file_name = self.file_name

        res_rf = {"trainCI": [], "trainCI.S.D.": [],
                  "testCI": [], "testCI.S.D.": []}

        if self.y_in_group:
            ro.r["source"]("computeRFSRC.R")
        else:
            ro.r["source"]("computeRFSRCinLatent.R")

        data_split, X, y = self._data_split()
        for train, test in data_split:
            Xtrain, Xtest, ytrain, ytest = X[train, :], X[test, :], y[train, :], y[test, :]

            # training_data = np.column_stack((Xtrain, ytrain))
            training_data = Xtrain
            # testing_data = np.column_stack((Xtest, ytest))
            testing_data = Xtest

            print("Xtrain: ", training_data.shape)
            print("Xtest: ", testing_data.shape)

            # transform numpy array to R object
            training_data_R = ro.r.matrix(training_data, nrow=training_data.shape[0],
                                          ncol=training_data.shape[1])

            testing_data_R = ro.r.matrix(testing_data, nrow=testing_data.shape[0],
                                         ncol=testing_data.shape[1])
            # rfRes = ro.r.computeRFSRC(training_data_R, testing_data_R)
            try:
                rfRes = ro.r.computeRFSRC(training_data_R, testing_data_R)
                res_rf["trainCI"].append(rfRes[0])
                res_rf["trainCI.S.D."].append(rfRes[1])
                res_rf["testCI"].append(rfRes[2])
                res_rf["testCI.S.D."].append(rfRes[3])
            except:
                print("Error in randomForestSRC!")
                res_rf["trainCI"].append(np.nan)
                res_rf["trainCI.S.D."].append(np.nan)
                res_rf["testCI"].append(np.nan)
                res_rf["testCI.S.D."].append(np.nan)

        res_rf_df = pd.DataFrame.from_dict(res_rf)
        foutstring_rf = "_Cindex_rf.csv"
        fout_rf = file_name + foutstring_rf
        res_rf_df.to_csv(fout_rf, index=False)
        mean_res_rf_df = res_rf_df.loc[:, ["trainCI", "testCI"]].mean(axis=0).T
        std_res_rf_df = res_rf_df.loc[:, ["trainCI", "testCI"]].std(axis=0).T

        return mean_res_rf_df.tolist() + std_res_rf_df.tolist()

""""
skf = KFold(n_splits=kFold)
for train, test in skf.split(X, y):

    Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]

    training_data = np.column_stack((Xtrain, ytrain))
    testing_data = np.column_stack((Xtest, ytest))
"""

