from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification

X, y = make_multilabel_classification(n_samples=1000, random_state=42, n_features=50, n_classes=4)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=700, test_size=300, random_state=42)

clf = TPOTClassifier(n_jobs=7, generations=5, population_size=4, verbosity=2, multi_output=True, )

clf.fit(X_train, y_train, )
clf.export('multiple_output_classifier.py')

print(clf.score(X_valid, y_valid))

y_pred = clf.predict(X_valid)
