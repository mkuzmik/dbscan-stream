import numpy as np
from scipy.spatial import ckdtree
from functools import reduce
import sys


class DBScanStream:
    """
    Stream optimized DBSCAN algorithm implementation based on
    https://github.com/jadwiga/Algorytm/blob/master/src/com/aware/plugin/template/Algorithms/ChangedDBSCAN.java

    API template based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30,
                 p=None, n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.labels_ = np.array([])
        self.search = None
        self.cluster_counter = 0

    def fit(self, X, sample_weight=None):
        # init all points as outliers
        self.labels_ = np.ones(len(X)) * -1

        # reset cluster counter
        self.cluster_counter = 0

        # init search tree
        self.search = NeighboursSearch(X, self.eps, self.min_samples)

        for point in X:
            neighbours_indices = self.search.get_neighbours(point)

            if len(neighbours_indices) >= self.min_samples:
                neighbours_clusters = map(
                    lambda neighbour_idx: self.labels_[neighbour_idx], neighbours_indices)
                neighbours_clusters_without_outliers = filter(
                    lambda neighbour_cluster: neighbour_cluster != -1, neighbours_clusters)
                min_neighbour_cluster = reduce(lambda a, b: a if a < b else b, neighbours_clusters_without_outliers,
                                               self.cluster_counter)
                self.cluster_counter = self.cluster_counter + 1 if min_neighbour_cluster == self.cluster_counter \
                    else self.cluster_counter
                self.labels_[neighbours_indices] = min_neighbour_cluster

        return self

    def partial_fit(self, X, sample_weight=None):
        # TODO
        return self

    def fit_predict(self, X, sample_weight=None):
        # TODO
        pass

    def get_params(self, deep=True):
        # TODO? - Dunno whether this needs to be implemented
        pass

    def set_params(self, **params):
        # TODO? - Dunno whether this needs to be implemented
        pass


class NeighboursSearch:
    def __init__(self, data, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.ckdtree = ckdtree.cKDTree(data, leafsize=100)

    def get_neighbours(self, point):
        """
        :param point: Point, which neighbours we are looking for
        :return: List of neighbours indices in self.data
        """
        neighbours_indices = self.ckdtree.query(point, k=self.min_samples, distance_upper_bound=self.eps)[1]
        return list(filter(lambda x: x != self.ckdtree.n, neighbours_indices))
