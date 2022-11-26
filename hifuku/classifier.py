from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.svm import SVC


@dataclass
class SVMDataset:
    positive_sample: List[np.ndarray]
    negative_sample: List[np.ndarray]

    @classmethod
    def from_xy(cls, X: List[np.ndarray], y: np.ndarray):
        assert y.dtype == bool
        X = np.array(X)  # type: ignore
        positive_sample = list(X[y])
        negative_sample = list(X[np.logical_not(y)])
        return cls(positive_sample, negative_sample)

    def __post_init__(self):
        assert len(self.positive_sample) > 0
        assert len(self.negative_sample) > 0

        pset = set([tuple(vec.tolist()) for vec in self.positive_sample])
        nset = set([tuple(vec.tolist()) for vec in self.negative_sample])
        assert pset.isdisjoint(nset)

    @property
    def n_positive(self) -> int:
        return len(self.positive_sample)

    @property
    def n_negative(self) -> int:
        return len(self.negative_sample)

    def get_labels(self, boolean: bool = False) -> np.ndarray:
        labels = [1.0] * self.n_positive + [0.0] * self.n_negative
        np_labels = np.array(labels)
        if boolean:
            return np_labels > 0.5
        else:
            return np_labels

    def visualize(self):
        all_sample = self.negative_sample + self.positive_sample
        pca = PCA(2)
        pca.fit(np.array(all_sample))

        x_posi = np.array(self.positive_sample)
        x_nega = np.array(self.negative_sample)
        x_posi_pca = pca.transform(x_posi)
        x_nega_pca = pca.transform(x_nega)

        plt.scatter(x_posi_pca[:, 0], x_posi_pca[:, 1], color="r")
        plt.scatter(x_nega_pca[:, 0], x_nega_pca[:, 1], color="b")
        plt.show()


@dataclass
class SVM:
    svm: SVC
    sigmoid_a: float
    sigmoid_b: float

    @classmethod
    def from_dataset(cls, dataset: SVMDataset, C=1.0) -> "SVM":
        svm = SVC(kernel="rbf", gamma="auto", probability=False, C=C)
        all_sample = dataset.positive_sample + dataset.negative_sample
        svm.fit(all_sample, dataset.get_labels())

        a, b = cls.fit_sigmoid(dataset, svm)
        return cls(svm, a, b)

    @staticmethod
    def fit_sigmoid(dataset: SVMDataset, svm: SVC):
        t_posi = (dataset.n_positive + 1.0) / (dataset.n_positive + 2.0)
        t_nega = 1.0 / (dataset.n_negative + 2.0)
        logicals = dataset.get_labels(boolean=True)
        t_vec = logicals * t_posi + ~logicals * t_nega

        all_sample = dataset.positive_sample + dataset.negative_sample
        f_vec = svm.decision_function(all_sample)

        def fun(params):
            A, B = params
            probs = 1.0 / (1.0 + np.exp(A * f_vec + B))
            values = -(t_vec * np.log(probs) + (1 - t_vec) * np.log(1 - probs))
            val = sum(values)
            return val

        sol = minimize(fun, [0.0, 0.0], method="BFGS")
        a, b = sol.x
        return a, b

    def predict(self, vec: np.ndarray) -> float:
        assert vec.ndim == 1
        val = self.svm.predict(np.expand_dims(vec, axis=0))
        return val.item()

    def predict_proba(self, vec: np.ndarray) -> float:
        fval = self.svm.decision_function(np.expand_dims(vec, axis=0))
        prob_vec = 1.0 / (1.0 + np.exp(self.sigmoid_a * fval + self.sigmoid_b))
        return prob_vec.item()
