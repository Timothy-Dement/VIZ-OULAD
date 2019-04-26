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
            
    for met in metrics:

        met_df = mod_master[mod_master['metric'] == met].copy(deep=True)

        met_df['clf+atbt'] = met_df['classifier'] + ' + ' + met_df['attributes']

        base_df = met_df[met_df['technique'] == 'base'].copy(deep=True)
        ae_df = met_df[met_df['technique'] == 'ae'].copy(deep=True)
        km_df = met_df[met_df['technique'] == 'kmeans'].copy(deep=True)
        pca_df = met_df[met_df['technique'] == 'pca'].copy(deep=True)

        ae_df['score'] = ae_df['score'] - base_df['score']
        km_df['score'] = km_df['score'] - base_df['score']
        pca_df['score'] = pca_df['score'] - base_df['score']

        sns.set(style='darkgrid')
        ae_plot = sns.barplot(x='clf+atbt', y='score', data=ae_df, color='#4682b4')
        ae_plot.set_title(f'{mod.upper()} : {met.upper()} \n Effect of Autoencoding', fontsize=15)

        plt.xticks(rotation=90)
        plt.xlabel('Classifier + Attribute Set')
        plt.ylabel('Score Difference')

        x = plt.gca().axes.get_xlim()
        plt.plot(x, (0,0), linewidth='2', color='black')

        ae_plot.figure.savefig(f'./charts/rq2_techniques/{mod.lower()}_{met}_ae.png')

        plt.clf()

        sns.set(style='darkgrid')
        km_plot = sns.barplot(x='clf+atbt', y='score', data=km_df, color='#4682b4')
        km_plot.set_title(f'{mod.upper()} : {met.upper()} \n Effect of K-Means Clustering', fontsize=15)

        plt.xticks(rotation=90)
        plt.xlabel('Classifier + Attribute Set')
        plt.ylabel('Score Difference')
        
        x = plt.gca().axes.get_xlim()
        plt.plot(x, (0,0), linewidth='2', color='black')

        ae_plot.figure.savefig(f'./charts/rq2_techniques/{mod.lower()}_{met}_km.png')

        plt.clf()

        sns.set(style='darkgrid')
        pca_plot = sns.barplot(x='clf+atbt', y='score', data=pca_df, color='#4682b4')
        pca_plot.set_title(f'{mod.upper()} : {met.upper()} \n Effect of PCA Feature Extraction', fontsize=15)

        plt.xticks(rotation=90)
        plt.xlabel('Classifier + Attribute Set')
        plt.ylabel('Score Difference')

        x = plt.gca().axes.get_xlim()
        plt.plot(x, (0,0), linewidth='2', color='black')

        ae_plot.figure.savefig(f'./charts/rq2_techniques/{mod.lower()}_{met}_pca.png')

        plt.clf()
        plt.close('all')


