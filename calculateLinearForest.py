import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score;
from sklearn.linear_model import LogisticRegression
from utils import fine_print

pd.set_option('display.max_rows', 150)
rand_seed = 42
data = pd.read_csv('2. Cup_IT_2022_Датасет_Data_Science.csv')
#normalized_data=(data-data.mean())/data.std()

columsToNormalize = ['cnt_checks_10_Мясная гастрономия','rto_std_11','rto_6','rto_12_Сыры','rto_12_Птица и изделия из птицы','rto_8','cnt_checks_9','rto_9_Мясная гастрономия','rto_10_Птица и изделия из птицы','cnt_checks_9_Рыба и рыбные изделия','rto_stddev_6_Рыба и рыбные изделия','rto_12_Мясная гастрономия','rto_stddev_8_Крупы и зерновые','rto_12','rto_6_Птица и изделия из птицы','rto_6_Рыба и рыбные изделия','cnt_checks_11','rto_stddev_7_Птица и изделия из птицы','rto_stddev_9_Сыры','cnt_checks_10_Овощи - Фрукты','cnt_checks_12_Птица и изделия из птицы','rto_6_Мясная гастрономия','rto_stddev_6_Овощи - Фрукты','cnt_checks_6_Мясная гастрономия','rto_stddev_6_Мясная гастрономия','cnt_checks_11_Сыры','cnt_checks_12_Овощи - Фрукты','rto_stddev_10_Птица и изделия из птицы','cnt_checks_6_Сыры','rto_stddev_10_Овощи - Фрукты','rto_12_Крупы и зерновые','rto_7_Птица и изделия из птицы','cnt_checks_12_Рыба и рыбные изделия','rto_stddev_10_Крупы и зерновые','rto_7_Крупы и зерновые','rto_7_Овощи - Фрукты','cnt_checks_10_Птица и изделия из птицы','rto_9_Рыба и рыбные изделия','cnt_checks_11_Крупы и зерновые','rto_stddev_11_Крупы и зерновые','rto_stddev_6_Крупы и зерновые','rto_stddev_12_Овощи - Фрукты','cnt_checks_11_Рыба и рыбные изделия','rto_stddev_9_Крупы и зерновые','rto_11_Крупы и зерновые','rto_stddev_10_Сыры','cnt_checks_12_Мясная гастрономия','rto_stddev_10_Мясная гастрономия','rto_stddev_8_Рыба и рыбные изделия','cnt_checks_9_Крупы и зерновые','rto_10_Сыры','rto_stddev_6_Сыры','cnt_checks_7_Сыры','cnt_checks_6_Птица и изделия из птицы','cnt_checks_6_Рыба и рыбные изделия','rto_8_Овощи - Фрукты','cnt_checks_10_Крупы и зерновые','rto_12_Овощи - Фрукты','rto_11_Мясная гастрономия','cnt_checks_12_Крупы и зерновые','rto_stddev_11_Мясная гастрономия','rto_stddev_12_Птица и изделия из птицы','rto_stddev_7_Крупы и зерновые','cnt_checks_6','cnt_checks_9_Овощи - Фрукты','cnt_checks_7_Крупы и зерновые','cnt_checks_8_Птица и изделия из птицы','rto_11_Птица и изделия из птицы','rto_stddev_11_Сыры','cnt_checks_7_Птица и изделия из птицы','rto_10_Овощи - Фрукты','rto_stddev_7_Мясная гастрономия','rto_9_Сыры','rto_10_Мясная гастрономия','rto_stddev_12_Рыба и рыбные изделия','cnt_checks_7','rto_7_Мясная гастрономия','cnt_checks_9_Мясная гастрономия','cnt_checks_7_Мясная гастрономия','rto_9_Овощи - Фрукты','rto_stddev_9_Птица и изделия из птицы','rto_std_9','rto_stddev_10_Рыба и рыбные изделия','rto_stddev_9_Рыба и рыбные изделия','cnt_checks_9_Сыры','rto_stddev_6_Птица и изделия из птицы','rto_10_Рыба и рыбные изделия','rto_12_Рыба и рыбные изделия','rto_10_Крупы и зерновые','cnt_checks_12','cnt_checks_10_Рыба и рыбные изделия','rto_stddev_9_Овощи - Фрукты','rto_11_Рыба и рыбные изделия','rto_6_Овощи - Фрукты','cnt_checks_10','rto_stddev_7_Овощи - Фрукты','cnt_checks_12_Сыры','rto_std_7','cnt_checks_8_Овощи - Фрукты','rto_stddev_8_Мясная гастрономия','rto_stddev_11_Овощи - Фрукты','cnt_checks_7_Овощи - Фрукты','rto_9_Птица и изделия из птицы','cnt_checks_6_Крупы и зерновые','cnt_checks_8_Крупы и зерновые','cnt_checks_7_Рыба и рыбные изделия','cnt_checks_11_Овощи - Фрукты','cnt_checks_8_Рыба и рыбные изделия','cnt_checks_11_Мясная гастрономия','rto_8_Мясная гастрономия','rto_10','rto_std_8','rto_stddev_11_Птица и изделия из птицы','rto_stddev_8_Птица и изделия из птицы','rto_stddev_12_Мясная гастрономия','rto_6_Сыры','rto_7','rto_std_6','cnt_checks_9_Птица и изделия из птицы','rto_stddev_11_Рыба и рыбные изделия','rto_9','rto_std_10','rto_stddev_8_Сыры','rto_8_Птица и изделия из птицы','rto_11_Овощи - Фрукты','cnt_checks_11_Птица и изделия из птицы','rto_7_Рыба и рыбные изделия','cnt_checks_6_Овощи - Фрукты','rto_9_Крупы и зерновые','rto_7_Сыры','rto_8_Крупы и зерновые','cnt_checks_8_Мясная гастрономия','rto_stddev_7_Сыры','rto_8_Сыры','rto_11','rto_std_12','cnt_checks_10_Сыры','rto_stddev_12_Сыры','rto_6_Крупы и зерновые','rto_stddev_7_Рыба и рыбные изделия','rto_stddev_8_Овощи - Фрукты','rto_8_Рыба и рыбные изделия','rto_11_Сыры','cnt_checks_8_Сыры','cnt_checks_8','rto_stddev_9_Мясная гастрономия','rto_stddev_12_Крупы и зерновые']
normalized_data = data
normalized_data[columsToNormalize]=normalized_data[columsToNormalize].apply(lambda x: x / x.max())
#normalized_data.fillna(0)
X = data.iloc[:, 2:]
y = data.iloc[:, 1]
X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
config = {
    #'n_estimators': [200, 500],
    #'max_features': ['auto', 'sqrt', 'log2'],
    #'max_depth' : [4,5,6,7,8],
    #'criterion' :['gini', 'entropy'],
    'class_weight': [{0: 122643, 1: 12418}]
}

scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
clfVer = 2
if(clfVer == 1):
    print('Forest in da house')
    clf = RandomForestClassifier(n_estimators=200,random_state=rand_seed, n_jobs=8)
elif(clfVer == 2):
    print("Its Logistic Regression in the casa")
    clf = LogisticRegression(random_state=rand_seed)


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rand_seed)
validator = GridSearchCV(clf, param_grid=config, scoring=scoring, n_jobs=4, cv=cv, refit='f1', verbose=2)

validator.fit(X, y)

fine_print(validator.cv_results_)