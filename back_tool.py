import pandas as pd
import  numpy as np
import os
import random

def processing_data(compete_type_filename, data_filename):
    file_names = []
    for root, dirs, files in os.walk(f'./{compete_type_filename}/{data_filename}'):
        for name in files:
            file_names.append (os.path.join(root, name))

    X_seqs = []
    y_targets = []
    # print(file_names)
    random.shuffle(file_names)
    for idx, file in enumerate(file_names):
        # print(file)
        X_seq_df = pd.read_csv(file)
        X_seq_df = X_seq_df.dropna(axis=1).drop(['ID'], axis=1)
        X_seq = X_seq_df.values
        y_targets.append(file.split('/')[-2])
        X_seq = X_seq.reshape(-1).tolist()
        X_seqs.append(X_seq)
        if idx > 100:break
    res_df = pd.DataFrame({'X_seqs':X_seqs, 'y_targets':y_targets}, index=None)
    res_df.sort_values(by=['y_targets'], inplace=True, ignore_index=True)
    res_df.to_pickle(f"./{compete_type_filename}/{data_filename}_data.pkl")


processing_data('模式识别数据集', 'train')
processing_data('模式识别数据集', 'validation')