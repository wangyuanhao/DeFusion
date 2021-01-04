#!usr/bin/python
# -*- coding: utf -*-

# evaluation

import pandas as pd

import rpy2.robjects as ro
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from itertools import compress
from sklearn.metrics.pairwise import cosine_similarity
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def generate_simlarity_mat(D):
    # return similarity matrix
    data_source_num = len(D)
    sim_mat = []
    for i in range(data_source_num):
        sim_mat.append(cosine_similarity(D[i]))

    return sim_mat


def draw_htmp_R(mat, group, fi, flag="sim"):
    # plot similarity matrix heatmap with R, group by group label
    ro.r['source']('dataHeatmapWithGroup.R')
    ro.r['source']('simHeatmapWithGroup.R')

    mat_R = ro.r.matrix(mat, nrow=mat.shape[0], ncol=mat.shape[1])

    group_R = ro.vectors.FloatVector(group)

    if flag == "sim":
        ro.r.simHeatmapWithGroup(mat_R, group_R, fi)
    else:
        ro.r.dataHeatmapWithGroup(mat_R, group_R, fi)


def survival_rate_stat(group, follow_up_time, follow_up_event):

    ro.r['source']('get_surv_rate.R')


    uq_group = list(set(group))
    surv_dict = dict()
    for group_label in uq_group:
        group_bool = [True if group_label==group[i] else False
                      for i, _ in enumerate(group)]

        surv_dict["group"+str(group_label)] = None

        this_follow_up_time =  list(compress(follow_up_time, group_bool))
        this_follow_up_event = list(compress(follow_up_event, group_bool))

        rfup_evnt = ro.vectors.IntVector(this_follow_up_event)
        rfup_time = ro.vectors.FloatVector(this_follow_up_time)

        surv_900 = ro.r.get_surv_rate(rfup_time, rfup_evnt, 900, 900)
        surv_1500 = ro.r.get_surv_rate(rfup_time, rfup_evnt, 1500, 1500)
        surv_dict["group"+str(group_label)] = list(surv_900) + list(surv_1500)

    return surv_dict


def surv_1500d_rate(group, follow_up_time, follow_up_event):

    death_with_1500d = [1 if (time < 1500 and event == 1) else 0
                        for time, event in zip(follow_up_time, follow_up_event)]
    surv_rate = 1 - sum(death_with_1500d) / len(group)
    return surv_rate


def surv_900d_rate(group, follow_up_time, follow_up_event):

    death_with_900d = [1 if (time < 900 and event == 1) else 0
                        for time, event in zip(follow_up_time, follow_up_event)]
    surv_rate = 1 - sum(death_with_900d) / len(group)
    return surv_rate


def Silhouette(X, label):
    return silhouette_score(X, labels=label)


def CaHar(X, label):
    return calinski_harabaz_score(X, labels=label)


def log_rank_test(group, follow_up_time, follow_up_event, fi, save_):
    # log rank test for survival analysis in different groups,
    # return p value of log rank test and save K-M curve in fi
    # Input:
    #   group: group label, list of length sample_num
    #   follow_up_time: follow up time, list of length sample_num
    #   follow_up_event: 0 means event did not happen, otherwise, 1,
    #                    e.g. 0=alive, 1=dead
    #   fi: file name for K-M curve image

    ro.r['source']('survAnalysis.R')

    rgroup = ro.vectors.StrVector(group)
    fac_rgroup = ro.vectors.FactorVector(rgroup)
    rfup_evnt = ro.vectors.IntVector(follow_up_event)
    rfup_time = ro.vectors.FloatVector(follow_up_time)

    pvalue = ro.r.surv_analysis(fac_rgroup, rfup_time, rfup_evnt, fi, save_)

    return pvalue[0]


def get_p_value(X, follow_up_time, follow_up_event, stage, result_path):

    cluster = [KMeans(n_clusters=X.shape[1], random_state=10),
               AgglomerativeClustering(n_clusters=X.shape[1])]
    cluster_str = ['Kmeans', 'AGC']
    cluster_score = {}
    pvalue_rcd = {}
    for i, clst in enumerate(cluster):
        group =clst.fit_predict(X)
        # compute log-rank test in different clustering
        fi = result_path + ' survival curves ' + cluster_str[i] + '.png'

        pvalue = log_rank_test(group, follow_up_time, follow_up_event, fi, save_=1)
        outcome_dict = {'group': group, 'follow_up_time': follow_up_time,
                        'follow_up_event': follow_up_event, 'stage': stage}
        outcome_df = pd.DataFrame.from_dict(outcome_dict)
        outcome_df.to_csv(result_path + 'cluster info '+ cluster_str[i] + '.csv', index=None)

        pvalue_rcd[cluster_str[i]] = pvalue
        cluster_score[cluster_str[i]+'_Silhouette'] = Silhouette(X, group)
        cluster_score[cluster_str[i]+'_CaHar'] = CaHar(X, group)

    return pvalue_rcd, cluster_score
