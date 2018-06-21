import numpy as np
import pandas as pd
import csv
import numpy

from sklearn.svm import *

from sklearn.preprocessing import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import *
from sklearn.model_selection import train_test_split

import os

DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/prova/")
df = pd.read_csv(DATA_PATH + 'data.csv')
print(df.diagnosis.unique())
# filtro usando Z-score
#df = df[(np.abs(stats.zscore(df.drop(['GENRE'], axis=1))) < 3).all(axis=1)]
y = df.diagnosis
X = df.drop(['diagnosis'], axis=1)
#print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
# info
print('Train dataset size:', X_train.shape)
print('Test dataset size:', X_test.shape)
print('TrainY dataset size:', y_train.shape)
print('TestY dataset size:', y_test.shape)

#ESCOLHA O KERNEL
#svc = LinearSVC(verbose=True)
svc = SVC(verbose=True)


print(cross_val_score(svc, X_train, y_train, cv=10))

#TREINO E TESTEs
svc.fit(X_train, y_train)
pred = svc.predict(X_test)

#print(pred)