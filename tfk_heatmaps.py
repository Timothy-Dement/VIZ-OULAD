import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp

rcp.update({'figure.autolayout': True})
rcp["figure.figsize"] = (7, 10)
plt.rcParams["axes.labelsize"] = 20

if not os.path.exists('./charts'):
    os.mkdir('./charts')

if not os.path.exists('./charts/tfk_heatmaps'):
    os.mkdir('./charts/tfk_heatmaps')

mod_paths = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

clf_paths = ['FF']
tch_paths = ['0', 'AE', 'KMEANS', 'PCA', 'SMOTE', 'SMOTE+AE']

metrics = ['accuracy', 'fscore', 'precision', 'recall']
attributes = ['asmt', 'stdnt', 'abd', 'abi', 'asmt_stdnt', 'asmt_abd', 'asmt_abi', 'stdnt_abd', 'stdnt_abi', 'asmt_stdnt_abd', 'asmt_stdnt_abi']

for mod in mod_paths:

    mod_master = pd.DataFrame(columns=['module', 'attributes', 'classifier', 'technique', 'metric', 'score'])

    for clf in clf_paths:
        for tch in tch_paths:

            clf_fname = clf.lower()
            tch_fname = 'base' if tch == '0' else tch.lower()

            df = pd.read_csv(f'./results/{clf}/{clf}-{tch}/{mod}_{clf_fname}-{tch_fname}_results.csv')
            
            mod_master = mod_master.append(df)

    mod_master['technique'] = mod_master['technique'].apply(lambda x: '' if x == 'base' else '+' + x)    
    mod_master['id'] = mod_master['classifier'] + mod_master['technique']

    for met in metrics:

        met_df = mod_master[mod_master['metric'] == met].copy(deep=True)

        met_df['attributes'] = pd.Categorical(met_df['attributes'], attributes)

        pivot = met_df.pivot(index='attributes', columns='id', values='score')

        hm = sns.heatmap(pivot, vmin=0, vmax=1, annot=True, fmt='.2f', cbar=False)

        hm.set_title(f'{mod.upper()} : {met.capitalize()}', fontsize=25)
        hm.set(xlabel='Classifier + Technique', ylabel='Attribute Set')
        hm.tick_params(labelsize=15)

        plt.xticks(rotation=90)

        hm.figure.savefig(f'./charts/tfk_heatmaps/{mod}-{met}_tfk_heatmap.png')

        plt.clf()
        plt.close('all')
