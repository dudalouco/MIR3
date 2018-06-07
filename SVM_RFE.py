import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

import os

DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/music-information-retrievel-3rd-edition/")
DATA_FILES = os.listdir(DATA_PATH)

print(DATA_FILES)
df = pd.read_csv(DATA_PATH + DATA_FILES[2])
print(df.GENRE.unique())
X = df.drop(['GENRE'], axis=1).values
y = df.GENRE.values

svc = SVC(kernel="linear")

rfecv = RFECV(estimator=svc, step=.7,
              cv=StratifiedKFold(3), scoring='accuracy')

rfecv.fit(X, y)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
