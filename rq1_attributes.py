import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp

rcp.update({'figure.autolayout': True})
rcp["figure.figsize"] = (5, 10)
plt.rcParams["axes.labelsize"] = 12

if not os.path.exists('./charts'):
    os.mkdir('./charts')

if not os.path.exists('./charts/rq1_attributes'):
    os.mkdir('./charts/rq1_attributes')

mod_paths = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

clf_paths = ['DT', 'FF', 'KNN', 'NB', 'NN', 'RF', 'SVM']
tch_paths = ['0']

metrics = ['accuracy', 'fscore', 'precision', 'recall']
attribute_order = ['asmt', 'stdnt', 'abd', 'abi', 'asmt_stdnt', 'asmt_abd', 'asmt_abi', 'stdnt_abd', 'stdnt_abi', 'asmt_stdnt_abd', 'asmt_stdnt_abi']

for mod in mod_paths:

    mod_master = pd.DataFrame(columns=['module', 'attributes', 'classifier', 'technique', 'metric', 'score'])

    for clf in clf_paths:
        for tch in tch_paths:

            clf_fname = clf.lower()
            tch_fname = 'base' if tch == '0' else tch.lower()

            df = pd.read_csv(f'./results/{clf}/{clf}-{tch}/{mod}_{clf_fname}-{tch_fname}_results.csv')
            
            mod_master = mod_master.append(df)

    for clf in [x.lower() for x in clf_paths]:

        clf_df = mod_master[mod_master['classifier'] == clf].copy(deep=True)

        clf_df['attributes'] = pd.Categorical(clf_df['attributes'], attribute_order)

        clf_pivot = clf_df.pivot(index='attributes', columns='metric', values='score')

        hm = sns.heatmap(clf_pivot, vmin=0, vmax=1, annot=True, fmt='.2f', cbar=False)
        hm.set_title(f'{mod.upper()} - {clf.upper()}\nAttribute Set Performance', fontsize=15)
        hm.set(xlabel='Metric', ylabel='Attribute Set')

        hm.figure.savefig(f'./charts/rq1_attributes/{mod}-{clf}_skl_heatmap.png')

        plt.clf()
        plt.close('all')
