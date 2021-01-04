#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from evalModule import *
from dataModule import rename_stage
import os
from sklearn.cluster import KMeans
from crossValidationCindexAUC_Class import crossValidationCindexAUC



def chck_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


proj = "TCGA_LIHC"

low_dim = 6
alpha = 1
gamma = 10
kFold = 3
rseed = 1


y_in_group = False
view = ["fpkm", "mirnas", "methy"]

# load data
data_dir = "./data/" + proj + "/"
data = [np.load(data_dir + v + ".npy") for v in view]

D = [d for d in data]

# load clinical info.
connected_clinical_file = data_dir + proj + "_proc_connected_clinical_info.csv"
connected_clinical_info = pd.read_csv(connected_clinical_file, header=0, index_col=None)

tumor_stage = connected_clinical_info['tumor_stage']
vital_status = connected_clinical_info['vital_status']
follow_up = connected_clinical_info['days_to_last_follow_up']

follow_up_time = list(follow_up)
val_index = list(follow_up >= 0)

follow_up_event = [1 if item == 'dead' else 0 for item in list(vital_status)]

stage = rename_stage(tumor_stage)
valid_patient_num = len(follow_up_time)


file_in = data_dir + "lowDim=%d" % low_dim + "_alpha=%s_gamma=%s" % (str(alpha), str(gamma)) +"_X.csv"
rbNMF_X = pd.read_csv(file_in, header=None)

group = KMeans(n_clusters=low_dim, random_state=0).fit_predict(rbNMF_X)

km_fi = "%s low_dim=%d" % (proj, low_dim) + "survival_curves.png"
group_dict = {"patientID": connected_clinical_info["submitter_id"].tolist(),
            "follow_up_event": follow_up_event, "follow_up_time": follow_up_time,
            "group_label": group}
group_df = pd.DataFrame.from_dict(group_dict, orient="columns")



valid_data = pd.concat([group_df.loc[:, ["follow_up_event", "follow_up_time"]], pd.DataFrame(rbNMF_X)], axis=1)

crossvalid = crossValidationCindexAUC(valid_data, kFold, y_in_group, data_dir + "%s_low_dim=%d"
                                      % (proj, low_dim), rseed)
res_cox = crossvalid.valid_coxCindex()
train_cox = np.mean(res_cox[0])
test_cox = np.mean(res_cox[1])

print("Average Harrell's C-index in training in %d-fold cross-validation: %.3f" % (kFold, train_cox))
print("Average Harrell's C-index in test in %d-fold cross-validation: %.3f" % (kFold, test_cox))



