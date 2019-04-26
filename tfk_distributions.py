import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp

rcp.update({'figure.autolayout': True})
rcp["figure.figsize"] = (10, 10)

if not os.path.exists('./charts'):
    os.mkdir('./charts')

if not os.path.exists('./charts/tfk_distributions'):
    os.mkdir('./charts/tfk_distributions')

mod_paths = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

clf_paths = ['FF']
tch_paths = ['0', 'AE', 'KMEANS', 'PCA', 'SMOTE', 'SMOTE+AE']

metrics = ['accuracy', 'fscore', 'precision', 'recall']

for mod in mod_paths:

    mod_master = pd.DataFrame(columns=['module', 'attributes', 'classifier', 'technique', 'metric', 'score'])

    for clf in clf_paths:
        for tch in tch_paths:

            clf_fname = clf.lower()
            tch_fname = 'base' if tch == '0' else tch.lower()

            df = pd.read_csv(f'./results/{clf}/{clf}-{tch}/{mod}_{clf_fname}-{tch_fname}_results.csv')
            
            mod_master = mod_master.append(df)

    for met in metrics:

        met_df = mod_master[mod_master['metric'] == met].copy(deep=True)

        leq_1 = len(met_df[(met_df['score'] >= 0.0) & (met_df['score'] <= 0.1)].index)
        leq_2 = len(met_df[(met_df['score'] > 0.1) & (met_df['score'] <= 0.2)].index)
        leq_3 = len(met_df[(met_df['score'] > 0.2) & (met_df['score'] <= 0.3)].index)
        leq_4 = len(met_df[(met_df['score'] > 0.3) & (met_df['score'] <= 0.4)].index)
        leq_5 = len(met_df[(met_df['score'] > 0.4) & (met_df['score'] <= 0.5)].index)
        leq_6 = len(met_df[(met_df['score'] > 0.5) & (met_df['score'] <= 0.6)].index)
        leq_7 = len(met_df[(met_df['score'] > 0.6) & (met_df['score'] <= 0.7)].index)
        leq_8 = len(met_df[(met_df['score'] > 0.7) & (met_df['score'] <= 0.8)].index)
        leq_9 = len(met_df[(met_df['score'] > 0.8) & (met_df['score'] <= 0.9)].index)
        leq_10 = len(met_df[(met_df['score'] > 0.9) & (met_df['score'] <= 1.0)].index)

        data = [['0.0 - 0.1', leq_1],
                ['0.1 - 0.2', leq_2],
                ['0.2 - 0.3', leq_3],
                ['0.3 - 0.4', leq_4],
                ['0.4 - 0.5', leq_5],
                ['0.5 - 0.6', leq_6],
                ['0.6 - 0.7', leq_7],
                ['0.7 - 0.8', leq_8],
                ['0.8 - 0.9', leq_9],
                ['0.9 - 1.0', leq_10]]

        counts = pd.DataFrame(columns=['bucket', 'count'], data=data)

        sns.set(style='darkgrid')

        if met == 'accuracy':
            color = '#4682b4'
        elif met == 'fscore':
            color = '#ff8c00'
        elif met == 'precision':
            color = '#90ee90'
        elif met == 'recall':
            color = '#b22222'

        mod_plot = sns.barplot(x='bucket', y='count', color=color, data=counts)
        mod_plot.set_title(f"{mod.upper()} : {met.capitalize()}", fontsize=50)
        mod_plot.tick_params(labelsize=25)
        mod_plot.set(ylim=(0,70))

        plt.ylabel('')
        plt.xlabel('')

        plt.xticks(rotation=90)

        mod_plot.figure.savefig(f'./charts/tfk_distributions/{mod}-{met}_tfk_distributions.png')

        plt.clf()
        plt.close('all')


