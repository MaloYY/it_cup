import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool, cv
from utils import save_config

pd.set_option('display.max_rows', 150)
ID = 1

rand_seed = 42

data = pd.read_csv('2. Cup_IT_2022_Датасет_Data_Science.csv')

X = data.iloc[:, 2:]
y = data.iloc[:, 1]

dataset = Pool(X, y)

catboost_conf = {"iterations": 7000,
          "depth": 4,
          "loss_function": "Logloss",
          "eval_metric": 'F1',
          'custom_metric' : ['Accuracy', 'Recall', 'Precision', 'AUC'],
          'early_stopping_rounds': 50,
          'l2_leaf_reg': 3,
          'use_best_model': True,
          'learning_rate': 1e-3,
          "verbose": 25,
          'random_state': 42,
          'simple_ctr' : 'Counter',
          'combinations_ctr' : 'Counter',
          #'store_all_simple_ctr' : True,
          #'ctr_leaf_count_limit' : 10,
          #'max_ctr_complexity': 6,
          #'grow_policy': 'Lossguide',
          'class_weights': {0: 0.09, 1: 0.91}
          }

CUR_CONFIG = catboost_conf
save_config(ID, CUR_CONFIG, 'CatBoost')
ID += 1

scores = cv(
    dataset, 
    params=CUR_CONFIG,
    seed=42,
    stratified=True,
    plot=True
)

scores[['test-F1-mean', 'test-Recall-mean', 'test-Precision-mean']].plot()