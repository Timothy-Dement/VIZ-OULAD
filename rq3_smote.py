import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp

rcp.update({'figure.autolayout': True})
rcp["figure.figsize"] = (25,10)
plt.rcParams["axes.labelsize"] = 20

if not os.path.exists('./charts'):
    os.mkdir('./charts')

if not os.path.exists('./charts/rq3_smote'):
    os.mkdir('./charts/rq3_smote')

mod_paths = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

clf_paths = ['DT', 'FF', 'KNN', 'NB', 'NN', 'RF', 'SVM']
tch_paths = ['0', 'AE', 'KMEANS', 'PCA', 'SMOTE', 'SMOTE+AE', 'SMOTE+KMEANS', 'SMOTE+PCA']

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

    mod_master['attributes'] = pd.Categorical(mod_master['attributes'], attribute_order)

    for met in metrics:

        met_df = mod_master[mod_master['metric'] == met].copy(deep=True)

        norm_df = met_df[met_df['technique'] == 'base'].copy(deep=True)
        norm_df = norm_df.append(met_df[met_df['technique'] == 'ae'].copy(deep=True))
        norm_df = norm_df.append(met_df[met_df['technique'] == 'kmeans'].copy(deep=True))
        norm_df = norm_df.append(met_df[met_df['technique'] == 'pca'].copy(deep=True))

        smote_df = met_df[met_df['technique'] == 'smote'].copy(deep=True)
        smote_df = smote_df.append(met_df[met_df['technique'] == 'smote+ae'].copy(deep=True))
        smote_df = smote_df.append(met_df[met_df['technique'] == 'smote+kmeans'].copy(deep=True))
        smote_df = smote_df.append(met_df[met_df['technique'] == 'smote+pca'].copy(deep=True))

        norm_df['technique'] = norm_df['technique'].apply(lambda x: '' if x == 'base' else ' + ' + x)
        norm_df['clf+tch'] = norm_df['classifier'] + norm_df['technique']
        
        norm_pivot = norm_df.pivot(columns='clf+tch', index='attributes', values='score')
        norm_hm = sns.heatmap(norm_pivot, vmin=0, vmax=1, annot=True, fmt='.2f', cbar=False)

        norm_hm.set_title(f'{mod.upper()} : {met.capitalize()}\nNon-SMOTE Performance', fontsize=25)
        norm_hm.set(ylabel='Attribute Set', xlabel='Classifier + Technique')
        norm_hm.tick_params(labelsize=15)

        norm_hm.figure.savefig(f'./charts/rq3_smote/{mod}-{met}_norm.png')

        plt.clf()

        smote_df['technique'] = smote_df['technique'].apply(lambda x: '' if x == 'base' else ' + ' + x)
        smote_df['clf+tch'] = smote_df['classifier'] + smote_df['technique']

        smote_pivot = smote_df.pivot(columns='clf+tch', index='attributes', values='score')
        smote_hm = sns.heatmap(smote_pivot, vmin=0, vmax=1, annot=True, fmt='.2f', cbar=False)

        smote_hm.set_title(f'{mod.upper()} : {met.capitalize()}\nSMOTE Performance', fontsize=25)
        smote_hm.set(ylabel='Attribute Set', xlabel='Classifier + Technique')
        smote_hm.tick_params(labelsize=15)

        smote_hm.figure.savefig(f'./charts/rq3_smote/{mod}-{met}_smote.png')

        plt.clf()
        plt.close('all')