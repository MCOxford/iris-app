#!/usr/bin/env python3

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import numpy as np


class MLModel(object):
    _models = {
        1: LogisticRegression,
        2: SVC,
        3: GaussianNB,
    }
    _need_rand_seed = [1, 2]

    def __init__(self):
        self._X_train, self._X_test = None, None
        self._y_train, self._y_test = None, None
        self._y_pred = None
        self._clf = None
        self._labels = None

    @property
    def clf(self):
        return self._clf

    @property
    def labels(self):
        return self._labels

    def split_data(self, X, y, ts=0.2, rs=0):
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=ts,
                                                                                        random_state=rs)
        else:
            raise ValueError('X and y must be numpy arrays')

    def fit_classifier(self, ml_id, rs=0):
        if ml_id not in self._models.keys():
            raise ValueError(f'{ml_id!r} not found')
        if self._X_train is None or self._y_train is None:
            self.split_data()
        if ml_id in self._need_rand_seed:
            self._clf = self._models[ml_id](random_state=rs)
        else:
            if rs > 0:
                print('GaussianNB does not require a random seed...')
            self._clf = self._models[ml_id]()
        self._clf.fit(self._X_train, self._y_train)
        self._labels = self.clf.classes_
        self._y_pred = self._clf.predict(self._X_test)

    def give_confusion_matrix(self):
        return confusion_matrix(self._y_test, self._y_pred)

    def give_score(self):
        return accuracy_score(self._y_test, self._y_pred)

    def give_report(self, names=None):
        if names is None:
            target = ['Class-' + str(i) for i in self._labels]
        else:
            target = [names[i] for i in range(len(names))]
        return classification_report(self._y_test, self._y_pred, target_names=target)


if __name__ == '__main__':
    pass
