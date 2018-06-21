import numpy as np
import pandas as pd
import csv

from sklearn.svm import *

from sklearn.preprocessing import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import *

import os

DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/music-information-retrievel-3rd-edition/")
DATA_FILES = os.listdir(DATA_PATH)

print(DATA_FILES)
df = pd.read_csv(DATA_PATH + DATA_FILES[1])
print(df.GENRE.unique())
y = df.GENRE.values
X = df.drop(['GENRE'], axis=1).values
test = pd.read_csv(DATA_PATH + DATA_FILES[2])

#kbest = SelectKBest(mutual_info_classif, k=90)

#X_new = kbest.fit_transform(X, y)
#X_norm = minmax_scale(X_new, axis=0)


tree = ExtraTreesClassifier()
tree = tree.fit(X, y)
smodel = SelectFromModel(tree, prefit=True)
X_new = smodel.transform(X)

X_new = robust_scale(X)



svc = LinearSVC(verbose=True)

#print(cross_val_score(svc, X_new, y, cv=5))



svc.fit(X_new, y)

#pred = svc.predict(test)
#pred = svc.predict(kbest.transform(test))
pred = svc.predict(smodel.transform(test))

print(pred)
pred_int = []
for ea in pred:
    if ea == "Pop":
        pred_int.append(5)
    elif ea == "Blues":
        pred_int.append(1)
    elif ea == "Jazz":
        pred_int.append(3)
    elif ea == "Classical":
        pred_int.append(2)
    elif ea == "Rock":
        pred_int.append(6)
    elif ea == "Metal":
        pred_int.append(4)

# print(pred_int)

preddf = pd.DataFrame(pred_int[:len(pred)], columns=['"Genres"'])
preddf.index = np.arange(1, len(preddf) + 1)
preddf.to_csv('submission.csv', index_label='"Id"', quoting=csv.QUOTE_NONE)
# # #


# # print(df.describe())
