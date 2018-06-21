import numpy as np
import pandas as pd
import csv

import os

from sklearn.preprocessing import *
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.datasets import make_classification
from scipy import stats

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier


from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

def pertileFilter(X, low=.05, high=.95):
    quant_df = X.quantile([low, high])
    X = X.apply(lambda x: x[(x >= quant_df.loc[low, x.name]) &
                            (x <= quant_df.loc[high, x.name])], axis=0)
    X.dropna(inplace=True)
    return X


DATA_PATH = os.path.expanduser(
    "~/.kaggle/competitions/music-information-retrievel-3rd-edition/")
df = pd.read_csv(DATA_PATH + 'genresTrain.csv')
test = pd.read_csv(DATA_PATH + 'genresTest2.csv')
print(df.GENRE.unique())
# filtro usando Z-score
#df = df[(np.abs(stats.zscore(df.drop(['GENRE'], axis=1))) < 3).all(axis=1)]
y = df.GENRE
X = df.drop(['GENRE'], axis=1)
print(X.shape)


tree = ExtraTreesClassifier()
tree = tree.fit(X, y)
smodel = SelectFromModel(tree, prefit=True)
X_new = smodel.transform(X)

X_new = robust_scale(X_new)


# build a classifier
clf = RandomForestClassifier(n_estimators=30)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_new, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_new, y)

print(grid_search.best_params_)
grid_search.estimator.fit(X_new,y)
pred = grid_search.estimator.predict(smodel.transform(test))

#pred = nn.predict(smodel.transform(test))

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