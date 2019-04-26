import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp

rcp.update({'figure.autolayout': True})
rcp["figure.figsize"] = (20, 5)
plt.rcParams["axes.labelsize"] = 12

if not os.path.exists('./charts'):
    os.mkdir('./charts')

if not os.path.exists('./charts/rq2_techniques'):
    os.mkdir('./charts/rq2_techniques')

mod_paths = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

clf_paths = ['DT', 'FF', 'KNN', 'NB', 'NN', 'RF', 'SVM']
tch_paths = ['0', 'AE', 'KMEANS', 'PCA']

metrics = ['accuracy', 'fscore', 'precision', 'recall']
attribute_order = ['asmt', 'stdnt', 'abd', 'abi', 'asmt_stdnt', 'asmt_abd', 'asmt_abi', 'stdnt_abd', 'stdnt_abi', 'asmt_stdnt_abd', 'asmt_stdnt_abi']

for mod in mod_paths:

    mod_master = pd.DataFrame(columns=['module', 'attributes', 'classifier', 'technique', 'metric', 'score'])

    for clf in clf_paths:
        for tch in tch_paths:

            clf_fname = clf.lower()
            tch_fname = 'base' if tch == '0' else tch.lower()

            clf_df = pd.read_csv(f'./results/{clf}/{clf}-{tch}/{mod}_{clf_fname}-{tch_fname}_results.csv')

            mod_master = mod_master.append(clf_df)

    acc_df = mod_master[mod_master['metric'] == 'accuracy'].copy(deep=True)
    fsc_df = mod_master[mod_master['metric'] == 'fscore'].copy(deep=True)
    pre_df = mod_master[mod_master['metric'] == 'precision'].copy(deep=True)
    rec_df = mod_master[mod_master['metric'] == 'recall'].copy(deep=True)

    print(mod.upper())
    print('ACC: ', acc_df['score'].max())
    print('FSC: ', fsc_df['score'].max())
    print('PRE: ', pre_df['score'].max())
    print('REC: ', rec_df['score'].max())
    print()