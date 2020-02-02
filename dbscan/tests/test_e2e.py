from dbscan.core import DBScanStream
import numpy as np


def test_should_not_create_any_cluster_from_outliers_only():
    # given
    dbscan = DBScanStream(eps=1, min_samples=5)
    X = np.array([[1, 2],
                  [2, 2]])

    # when
    clustering = dbscan.fit(X)

    # then (outliers label is -1)
    assert clustering.labels_ is np.array([-1, -1])


