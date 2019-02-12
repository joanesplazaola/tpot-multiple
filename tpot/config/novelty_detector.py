# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

# Check the TPOT documentation for information on the structure of config dicts

anomaly_detector_config_dict = {

	# Anomaly detectors
	'sklearn.ensemble.IsolationForest': {
	},
	'sklearn.covariance.EllipticEnvelope': {
	},

	'sklearn.svm.OneClassSVM': {

		'gamma': np.logspace(-3, 2, 6),
		'nu': np.linspace(0.01, 1, 5),
		'kernel': ['rbf', 'sigmoid', 'linear'],

	},

	# Preprocesssors
	'sklearn.preprocessing.Binarizer': {
		'threshold': np.arange(0.0, 1.01, 0.05)
	},

	'sklearn.decomposition.FastICA': {
		'tol': np.arange(0.0, 1.01, 0.05)
	},

	'sklearn.cluster.FeatureAgglomeration': {
		'linkage': ['ward', 'complete', 'average'],
		'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
	},

	'sklearn.preprocessing.MaxAbsScaler': {
	},

	'sklearn.preprocessing.MinMaxScaler': {
	},

	'sklearn.preprocessing.Normalizer': {
		'norm': ['l1', 'l2', 'max']
	},

	'sklearn.kernel_approximation.Nystroem': {
		'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
		'gamma': np.arange(0.0, 1.01, 0.05),
		'n_components': range(1, 11)
	},

	'sklearn.decomposition.PCA': {
		'svd_solver': ['randomized'],
		'iterated_power': range(1, 11)
	},

	'sklearn.preprocessing.PolynomialFeatures': {
		'degree': [2],
		'include_bias': [False],
		'interaction_only': [False]
	},

	'sklearn.kernel_approximation.RBFSampler': {
		'gamma': np.arange(0.0, 1.01, 0.05)
	},

	'sklearn.preprocessing.RobustScaler': {
	},

	'sklearn.preprocessing.StandardScaler': {
	},

	'tpot.builtins.ZeroCount': {
	},

	'tpot.builtins.OneHotEncoder': {
		'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
		'sparse': [False],
		'threshold': [10]
	},

	'sklearn.feature_selection.SelectFwe': {
		'alpha': np.arange(0, 0.05, 0.001),
		'score_func': {
			'sklearn.feature_selection.f_classif': None
		}
	},

	'sklearn.feature_selection.SelectPercentile': {
		'percentile': range(1, 100),
		'score_func': {
			'sklearn.feature_selection.f_classif': None
		}
	},

	'sklearn.feature_selection.VarianceThreshold': {
		'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
	},

	'sklearn.feature_selection.RFE': {
		'step': np.arange(0.05, 1.01, 0.05),
		'estimator': {
			'sklearn.ensemble.ExtraTreesClassifier': {
				'n_estimators': [100],
				'criterion': ['gini', 'entropy'],
				'max_features': np.arange(0.05, 1.01, 0.05)
			}
		}
	},

	'sklearn.feature_selection.SelectFromModel': {
		'threshold': np.arange(0, 1.01, 0.05),
		'estimator': {
			'sklearn.ensemble.ExtraTreesClassifier': {
				'n_estimators': [100],
				'criterion': ['gini', 'entropy'],
				'max_features': np.arange(0.05, 1.01, 0.05)
			}
		}
	}

}
