class DBScanStream:
    """
    Stream optimized DBSCAN algorithm implementation based on
    https://github.com/jadwiga/Algorytm/blob/master/src/com/aware/plugin/template/Algorithms/ChangedDBSCAN.java
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

    def fit(self, X, sample_weight=None):
        # TODO 
        pass

    def partial_fit(self, X, sample_weight=None):
        # TODO
        pass

    def fit_predict(self, X, sample_weight=None):
        # TODO
        pass

    def get_params(self, deep=True):
        # TODO? - Dunno whether this needs to be implemented
        pass

    def set_params(self, **params):
        # TODO? - Dunno whether this needs to be implemented
        pass
