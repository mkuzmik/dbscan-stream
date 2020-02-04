import numpy as np
from sklearn.neighbors import KDTree
from functools import reduce


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
        self.__validate_metric()

        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

        self.labels_ = np.array([])
        self.search = None
        self.cluster_counter = 0
        self.sample_weight = np.array([])

    def fit(self, X, sample_weight=None):
        self.__search_neighbourhood(X, sample_weight)
        for point in X:
            self.__partial_fit_single_point(point)
        return self

    def partial_fit(self, X, sample_weight=None):
        self.__update_neighbourhood(X, sample_weight)
        for point in X:
            self.__partial_fit_single_point(point)
        return self

    def fit_predict(self, X, sample_weight=None):
        self.fit(X, sample_weight)
        return self.labels_

    def __search_neighbourhood(self, X, sample_weight=None):
        self.cluster_counter = 0
        self.labels_ = np.ones(len(X)) * -1
        self.search = NeighboursSearch(X, self.eps, self.min_samples, self.metric)

        assert sample_weight is None or len(sample_weight) == len(X)
        self.sample_weight = sample_weight if sample_weight is not None else np.ones(len(X))

    def __update_neighbourhood(self, X, sample_weight):
        if self.labels_ is None or self.search is None:
            self.__search_neighbourhood(X, sample_weight)
        else:
            self.labels_ = np.concatenate([self.labels_, np.ones(len(X)) * -1])
            self.search = NeighboursSearch(np.concatenate([self.search.get_data(), X]), self.eps, self.min_samples,
                                           self.metric)
            self.sample_weight = np.concatenate([self.sample_weight,
                                                 sample_weight if sample_weight is not None else np.ones(len(X))])

    def __partial_fit_single_point(self, X):
        neighbours_indices = self.search.get_neighbours(X)
        neighbours_weights = self.sample_weight[neighbours_indices]
        if sum(neighbours_weights) >= self.min_samples:
            neighbours_clusters = [self.labels_[neighbour_idx] for neighbour_idx in neighbours_indices]
            neighbours_clusters_without_outliers = filter(
                lambda cluster_label: cluster_label != -1,
                neighbours_clusters
            )
            min_neighbour_cluster = reduce(
                lambda a, b: a if a < b else b,
                neighbours_clusters_without_outliers,
                self.cluster_counter
            )

            if min_neighbour_cluster == self.cluster_counter:
                self.cluster_counter += 1

            self.labels_[neighbours_indices] = min_neighbour_cluster

    def __validate_metric(self):
        allowed_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean',
                           'mahalanobis']
        assert self.metric in allowed_metrics, 'Unknown metric value: %s' % self.metric


class NeighboursSearch:
    def __init__(self, data, eps, min_samples, metric):
        self.eps = eps
        self.min_samples = min_samples
        self.kdtree = KDTree(np.array(data), leaf_size=100, metric=metric)

    def get_data(self):
        return list(self.kdtree.data)

    def get_neighbours(self, point):
        """
        :param point: Point, which neighbours we are looking for
        :return: List of neighbours indices in self.data
        """
        neighbours = self.kdtree.query_radius([point], r=self.eps)
        return neighbours[0]
