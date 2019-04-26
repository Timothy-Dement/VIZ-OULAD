import copy as cp
import numpy as np
import pandas as pd
import statistics as sx

clf_paths = ['DT', 'FF', 'KNN', 'NB', 'NN', 'RF', 'SVM']
tch_paths = ['0', 'AE', 'KMEANS', 'PCA', 'SMOTE', 'SMOTE+AE', 'SMOTE+KMEANS', 'SMOTE+PCA']
mod_paths = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg']

tch_paths =['0']

metrics = ['accuracy', 'fscore', 'precision', 'recall']

# headers = ['attributes',
#            'classifier',
#            'technique',
#            'metric',
#            'aaa-score',
#            'bbb-score',
#            'ccc-score',
#            'ddd-score',
#            'eee-score',
#            'fff-score',
#            'ggg-score']

# master_df = pd.DataFrame(columns=headers)

for clf in clf_paths:
    for tch in tch_paths:
        for mod in mod_paths:
            tch_label = 'base' if tch == '0' else tch.lower()
            df = pd.read_csv(f'./results/{clf}/{clf}-{tch}/{mod}_{clf.lower()}-{tch_label}_results.csv')
            print(list(df))
