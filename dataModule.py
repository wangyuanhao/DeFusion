#!usr/bin/python
# -*- coding: utf-8 -*-

# preprocessing and normalizing

import numpy as np
import pandas as pd
from itertools import compress
import time

def drop_zeros(X, dim, percentage):
    # drop sample or feature with too many zeros
    # Input:
    #   X: data frame, sample x feature
    #   dim: direction of dropping zeros
    #        dim = 1, drop sample
    #        dim = 0, drop feature
    #   percentage: threshold of percentage of zeros
    # Output:
    #   procX: data frame with fewer zeros feature or sample, sample x feature
    #   rm_lst: samples or features removed
    sample_num, feature_num = X.shape
    features = list(X.columns)
    samples = list(X.index)
    if dim == 0:
        zeros_perct = X.apply(lambda x: x.value_counts().get(0, 0), axis=0) / (1.0*sample_num)
        procX = X.loc[:, zeros_perct <= percentage]
        rm_lst = list(compress(features, zeros_perct > percentage))

    elif dim == 1:
        zeros_perct = X.apply(lambda x: x.value_counts().get(0, 0), axis=1) / (1.0*feature_num)
        procX = X.loc[zeros_perct <= percentage, :]
        rm_lst = list(compress(samples, zeros_perct > percentage))

    else:
        procX = []
        rm_lst = []

    return procX, rm_lst


def drop_mean(X, threshold):
    # drop feature with low mean
    # Input:
    #   X: data frame, sample x feature
    #   threshold: mean value below threshold will be regarded
    #             as low mean
    # Output:
    #   procX: data frame without low-mean features, sample x feature
    #   rm_lst: features removed

    features = list(X.columns)
    boolean = X.mean(axis=0) > threshold
    procX = X.loc[:, boolean]

    rm_lst = list(compress(features, list(X.mean(axis=0) <= threshold)))

    return procX, rm_lst


def drop_var(X, threshold):
    # drop feature with low variance
    # Input:
    #   X: data frame, sample x feature
    #   threshold: variance below threshold will be regarded
    #             as low variance
    # Output:
    #   procX: data frame without low-variance features, sample x feature
    #   rm_lst: features removed

    features = list(X.columns)
    boolean = X.var(axis=0) > threshold
    procX = X.loc[:, boolean]

    rm_lst = list(compress(features, list(X.var(axis=0) <= threshold)))

    return procX, rm_lst


def drop_median_absoulte_dev(X, threshold):
    # median absolute deviation(MAD) for gene filter
    # MAD = median|Xi - median(Xi)|
    # Input:
    #   X: data frame, sample x feature
    #   threshold: MAD below threshold will be regarded as low MAD
    # Output:
    #   procX: data with without low MAD, sample x feature
    #   rm_lst: features removed
    features = list(X.columns)
    median_diff = abs(X - X.median(axis=0))
    blean = median_diff.median(axis=0) > threshold
    procX = X.loc[:, blean]

    rm_lst = list(compress(features, list(median_diff.median(axis=0) <= threshold)))

    return procX, rm_lst


def normal_median_normalized(X, label_info):
    # centered on the median of available normal sample
    # Input:
    #   X: data frame, sample x feature
    #   label_info: data frame x 1, label information
    #               of every sample
    # Output:
    #   procX: centered feature on the median of available
    #           normal sample

    # label_info = list(label_info)
    norm_boolean = [True if 'Normal' in info else False for info in label_info]
    norm_sample = X.iloc[norm_boolean, :]
    norm_sample_median = norm_sample.median(axis=0)

    procX = X - norm_sample_median

    return procX


def mean_std_norm(X):
    # normalize X with mean and standard deviation
    # Input:
    #   X: data frame, sample x feature
    # Output:
    #   procX: normalized X, sample x feature
    procX = (X - X.mean(axis=0)) / X.std(axis=0)

    if procX.isnull().values.any():
        procX = procX.fillna(0)

    return procX


def log_ratio_based_normal(X, label_info):
    # normalize X with log ratio with all sample vs normal sample
    # using median
    # Input:
    #   X: data frame, sample x feature
    #   label_info: data frame, sample x 1, label
    #               information of every sample
    # Output:
    #   procX: normalized X with log ratio
    # label_info = list(label_info)
    norm_boolean = [True if 'Normal' in info else False for info in label_info]
    norm_sample = X.iloc[norm_boolean, :]
    norm_sample_median = norm_sample.median(axis=0)

    procX = np.log((X+1)/(norm_sample_median + 1))

    if procX.isnull().values.any():
        procX = procX.fillna(0)

    return procX


def zero_one_standarize(X):
    # map feature value to [0, 1]
    # Input:
    #   X: data frame, sample x feature
    # Output:
    #   procX: data frame with all features in [0, 1]

    nrow, ncol = X.shape
    procX = X.apply(lambda x: (x - x.min())/max((x.max()-x.min()), 1e-25), axis=0)
    return procX


def preprocessing(dir, proj, omics_type, logfile):
    # preprocessing, dropping features and normalizing
    # Input:
    #   dir: data location
    #   proj: project tag
    #   omic_type: omic type of data source
    #   logfile: txt file for recording log
    # Output:
    #   proc_df: postprocessing data frame, sample x feature

    logfile = open(logfile, 'a+')

    exp_file = dir + proj + '_' + omics_type + '.txt'
    clinical_file = dir + proj + '_clinical_info_' + omics_type + '.csv'

    # log
    logfile.writelines('*'*60+'\n')
    logfile.writelines('processing ' + omics_type + ' at '+time.asctime()+'\n')

    if omics_type == 'post_nocnv':
        exp_data = pd.read_csv(exp_file, sep='\t', index_col=0).T.iloc[5:-1, :]
        exp_data = exp_data.fillna(0)
        #exp_data = exp_data
        mean_threshold = 0
        var_threshold = 0

    if omics_type in ['compressing_27_450', 'compressing_450']:
        exp_data = pd.read_csv(exp_file, sep='\t', index_col=0).T.iloc[4:-1,:]
        exp_data = exp_data.fillna(0)
        #file_ID = list(exp_data.index)
        mean_threshold = 0.1
        var_threshold = 0.05

    if omics_type in ['.FPKM', '.mirnas']:
        exp_data = pd.read_csv(exp_file, sep='\t', index_col=0)
        exp_data = exp_data.fillna(0)
        # file_ID = list(exp_data.index)
        mean_threshold = 0.1
        var_threshold = 0.01

    # fill nan
    if exp_data.isnull().values.any():
        exp_data.fillna(0)

    # exp_data = pd.read_csv(exp_file, sep='\t', index_col=0)
    clinical = pd.read_csv(clinical_file)

    id = list(map(lambda x: clinical[clinical['file_id'] == x].index.tolist()[0],
                  list(exp_data.index)))

    # annotate every record in expressive data with sample_type clinical
    sample_type = list(clinical['sample_type'].iloc[id])


    # drop features
    proc_exp_data, rm_lst = drop_zeros(exp_data, 0, 0.2)

    # log
    logfile.writelines('drop features with zeros-percentage greater than '+str(0.2)+'\n')
    logfile.writelines(str(len(rm_lst))+' features removed\n')
    logfile.writelines('size of obtained data frame '+str(proc_exp_data.shape)+'\n')

    proc_exp_data = zero_one_standarize(proc_exp_data)
    # log
    logfile.writelines('\nmapping all feature values into [0, 1]\n')

    proc_exp_drop_mean, rm_lst_drop_mean = drop_mean(proc_exp_data, mean_threshold)

    # log
    logfile.writelines('\ndrop features with mean below '+str(mean_threshold)+'\n')
    logfile.writelines(str(len(rm_lst_drop_mean))+' features removed\n')
    logfile.writelines('size of obtained data frame: '+str(proc_exp_drop_mean.shape)+'\n')

    proc_exp_drop_var, rm_lst_drop_var = drop_var(proc_exp_drop_mean, var_threshold)

    # log
    logfile.writelines('\ndrop features with variance smaller than '+str(var_threshold)+'\n')
    logfile.writelines(str(len(rm_lst_drop_var))+' features removed\n')
    logfile.writelines('size of obtained data frame: '+str(proc_exp_drop_var.shape)+'\n')

    proc_exp_drop_mad, rm_lst_drop_mad = drop_median_absoulte_dev(proc_exp_drop_var, 0.00)

    # log
    logfile.writelines('\ndrop features with median absolute derivation below '+str(0.00)+'\n')
    logfile.writelines(str(len(rm_lst_drop_mad))+' features removed\n')
    logfile.writelines('size of obtained data frame: '+str(proc_exp_drop_mad.shape)+'\n')

    # normalize with normal samples
    # log
    logfile.writelines('\nmedian centralizing with normal samples\n')
    proc_exp_norm = normal_median_normalized(proc_exp_drop_mad, sample_type)

    # log
    # logfile.writelines('\nlog ratio normalizing with normal sample')
    # proc_exp_log_norm = log_ratio_based_normal(proc_exp_norm, sample_type)

    # standarization
    # log
    # logfile.writelines('\nmean and standard variance standarizing\n')
    # proc_exp_norm_standarized = mean_std_norm(proc_exp_norm)

    # plt.imshow(proc_exp_norm_standarized.values)
    # plt.show()
    # print('')

    logfile.close()

    return proc_exp_norm


def rename_stage(stage):
    # combine stage xxa, xxb, and xxc
    # Input:
    #   stage: list of stage
    # Output:
    #   re_stage: list of combine stage
    re_stage = []
    for item in stage:
        if item in ['stage i', 'stage ia', 'stage ib', 'stage ic']:
            re_stage.append('stage i')
        elif item in ['stage ii', 'stage iia', 'stage iib', 'stage iic']:
            re_stage.append('stage ii')
        elif item in ['stage iii', 'stage iiia', 'stage iiib', 'stage iiic']:
            re_stage.append('stage iii')
        elif item in ['stage iv', 'stage iva', 'stage ivb', 'stage ivc']:
            re_stage.append('stage iv')
        elif item in ['stage x', 'stage xa', 'stage xb', 'stage xc']:
            re_stage.append('stage x')
        else:
            re_stage.append('uknwn')

    return re_stage