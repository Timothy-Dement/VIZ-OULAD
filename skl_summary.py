import os

import pandas as pd

if not os.path.exists('./charts'):
    os.mkdir('./charts')

if not os.path.exists('./charts/skl_heatmaps'):
    os.mkdir('./charts/skl_heatmaps')

mod_paths = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

clf_paths = ['DT', 'KNN', 'NB', 'NN', 'RF', 'SVM']
tch_paths = ['0', 'AE', 'KMEANS', 'PCA', 'SMOTE']

metrics = ['accuracy', 'fscore', 'precision', 'recall']
attributes = ['asmt', 'stdnt', 'abd', 'abi', 'asmt_stdnt', 'asmt_abd', 'asmt_abi', 'stdnt_abd', 'stdnt_abi', 'asmt_stdnt_abd', 'asmt_stdnt_abi']

master = pd.DataFrame(columns=['module', 'attributes', 'classifier', 'technique', 'metric', 'score'])

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

        max_score = met_df['score'].max()
        print('\n=========> {0} : {1} : {2:.4f}\n'.format(mod.upper(), met.upper(), max_score))

        win_df = met_df[met_df['score'] == max_score].copy(deep=True)

        master = master.append(win_df)

        num_wins = len(win_df.index)

        atbt_labels = list(win_df['attributes'].value_counts().index)
        atbt_counts = list(win_df['attributes'].value_counts())

        clf_labels = list(win_df['classifier'].value_counts().index)
        clf_counts = list(win_df['classifier'].value_counts())

        tch_labels = list(win_df['technique'].value_counts().index)
        tch_counts = list(win_df['technique'].value_counts())

        print('\t+------------+\n\t| ATTRIBUTES |\n\t+------------+')
        for i, v in enumerate(atbt_counts):
            print('\t{0}\t{1:.2f}\t{2}'.format(v, v / num_wins * 100, atbt_labels[i]))

        print('\n\t+-------------+\n\t| CLASSIFIERS |\n\t+-------------+')
        for i, v in enumerate(clf_counts):
            print('\t{0}\t{1:.2f}\t{2}'.format(v, v / num_wins * 100, clf_labels[i]))
        
        print('\n\t+------------+\n\t| TECHNIQUES |\n\t+------------+')
        for i, v in enumerate(tch_counts):
            print('\t{0}\t{1:.2f}\t{2}'.format(v, v / num_wins * 100, tch_labels[i]))

for met in metrics:

    met_df = master[master['metric'] == met].copy(deep=True)

    print('\n=========> ALL : {0}\n'.format(met.upper()))

    for score in met_df['score'].sort_values(ascending=False).unique():
        print('\t{0:.4f}'.format(score))

    num_items = len(met_df.index)

    atbt_labels = list(met_df['attributes'].value_counts().index)
    atbt_counts = list(met_df['attributes'].value_counts())

    clf_labels = list(met_df['classifier'].value_counts().index)
    clf_counts = list(met_df['classifier'].value_counts())

    tch_labels = list(met_df['technique'].value_counts().index)
    tch_counts = list(met_df['technique'].value_counts())

    print('\n\t+------------+\n\t| ATTRIBUTES |\n\t+------------+')
    for i, v in enumerate(atbt_counts):
        print('\t{0}\t{1:.2f}\t{2}'.format(v, v / num_items * 100, atbt_labels[i]))

    print('\n\t+-------------+\n\t| CLASSIFIERS |\n\t+-------------+')
    for i, v in enumerate(clf_counts):
        print('\t{0}\t{1:.2f}\t{2}'.format(v, v / num_items * 100, clf_labels[i]))
    
    print('\n\t+------------+\n\t| TECHNIQUES |\n\t+------------+')
    for i, v in enumerate(tch_counts):
        print('\t{0}\t{1:.2f}\t{2}'.format(v, v / num_items * 100, tch_labels[i]))

print()