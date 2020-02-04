import numpy as np

from dbscan.core import DBScanStream


def test_outliers_only():
    # given
    dbscan = DBScanStream(eps=1, min_samples=5)
    X = np.array([[1, 2],
                  [2, 2]])

    # when
    clustering = dbscan.fit(X)

    # then (outliers label is -1)
    assert np.array_equal(clustering.labels_, np.array([-1, -1]))


def test_one_cluster():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=3)
    X = np.array([[1, 1],
                  [2, 2],
                  [0, 1]])

    # when
    clustering = dbscan.fit(X)

    # then (outliers label is -1)
    assert np.array_equal(clustering.labels_, np.array([0, 0, 0]))


def test_two_clusters():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=2)
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7]])

    # when
    clustering = dbscan.fit(X)

    # then (outliers label is -1)
    assert np.array_equal(clustering.labels_, np.array([0, 0, 1, 1]))


def test_two_clusters_with_outliers():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=2)
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7],
                  [-1, -2],
                  [56, 34]])

    # when
    clustering = dbscan.fit(X)

    # then (outliers label is -1)
    assert np.array_equal(clustering.labels_, np.array([0, 0, 1, 1, -1, -1]))


def test_partial_fit_of_two_clusters_with_outliers():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=2)
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7],
                  [-1, -2],
                  [56, 34]])

    # when
    for point in X:
        dbscan.partial_fit([point])

    # then (outliers label is -1)
    assert np.array_equal(dbscan.labels_, np.array([0, 0, 1, 1, -1, -1]))


def test_partial_fit_in_batches_of_two_clusters_with_outliers():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=2)
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7],
                  [-1, -2],
                  [56, 34]])

    # when
    for i in range(0, len(X), 2):
        dbscan.partial_fit(X[i:i+2])

    # then (outliers label is -1)
    assert np.array_equal(dbscan.labels_, np.array([0, 0, 1, 1, -1, -1]))


if __name__ == '__main__':
    test_two_clusters_with_outliers()