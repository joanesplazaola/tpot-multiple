import pandas as pd
import numpy as np
from tpot.tpot import TPOTNoveltyDetector


def get_target_and_factors(df, target_col):
	factors = df.loc[:, df.columns != target_col]
	target = df[target_col]
	return factors, target


def split_vals(df, n): return df[:n], df[n:]


dataset = pd.read_csv('/home/joanes/GBL/data/Novelty/creditcard.csv')  # .drop('Time', axis=1)
dataset = dataset[:10_000]
dataset['Amount'] = np.log(dataset['Amount'] + 1)
dataset['Time'] = np.log(dataset['Time'] + 1)
dataset[dataset['Class'] == 1] = -1
dataset[dataset['Class'] == 0] = 1

train_ratio = 0.7
train_size = int(dataset.shape[0] * train_ratio)

train, test = split_vals(dataset, train_size)

train_normal = train[train['Class'] == 1]

X_train, y_train = get_target_and_factors(train_normal, 'Class')
X_test, y_test = get_target_and_factors(test, 'Class')

ad = TPOTNoveltyDetector(generations=10, population_size=5, n_jobs=7, verbosity=2)

ad.fit(X_train, features_valid=X_test, target_valid=y_test)

ad.export('anomaly_pipeline.py')
