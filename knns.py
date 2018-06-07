import numpy as np
import pandas as pd
import csv

from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.model_selection import cross_val_score

import os

DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/music-information-retrievel-3rd-edition/")
DATA_FILES = os.listdir(DATA_PATH)

print(DATA_FILES)
df = pd.read_csv(DATA_PATH + DATA_FILES[2])
print(df.GENRE.unique())
y = df.GENRE.values
X = df.drop(['GENRE'], axis=1).values
test = pd.read_csv(DATA_PATH + DATA_FILES[0])

# for nn in range(3, 30):
#     knn = KNN(n_neighbors=nn)
#     # knn.fit(X, y)
#     scores = cross_val_score(knn, X, y, cv=5)
#     print('scores de ' + str(nn))
#     print(scores)

knn = KNN(n_neighbors=8)
knn.fit(X, y)

pred = knn.predict(test)

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
preddf.to_csv('predicted.csv', index_label='"Id"', quoting=csv.QUOTE_NONE)
# #


# print(df.describe())
