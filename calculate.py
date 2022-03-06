import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score;
print('Forest in da house')

pd.set_option('display.max_rows', 150)
rand_seed = 42
data = pd.read_csv('2. Cup_IT_2022_Датасет_Data_Science.csv')
X = data.iloc[:, 2:]
y = data.iloc[:, 1]
X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
config = {
    #'n_estimators': [200, 500],
    #'max_features': ['auto', 'sqrt', 'log2'],
    #'max_depth' : [4,5,6,7,8],
    #'criterion' :['gini', 'entropy'],
    'class_weight': [None]#, {0: 122643, 1: 12418}]
}

scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
clf = RandomForestClassifier(n_estimators=100,random_state=rand_seed, n_jobs=8)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rand_seed)
validator = GridSearchCV(clf, param_grid=config, scoring=scoring, n_jobs=4, cv=cv, refit='f1', verbose=2)

validator.fit(X, y)

for x in validator.cv_results_:
    print(x[])
