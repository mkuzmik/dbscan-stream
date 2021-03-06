import numpy as np
import pytest
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
        dbscan.partial_fit(X[i:i + 2])

    # then (outliers label is -1)
    assert np.array_equal(dbscan.labels_, np.array([0, 0, 1, 1, -1, -1]))


def test_two_clusters_with_same_sample_weights():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=4)
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7],
                  [-1, -2],
                  [56, 34]])

    sample_weight = np.array([2, 2, 2, 2, 2, 2])

    # when
    clustering = dbscan.fit(X, sample_weight)

    # then (outliers label is -1)
    assert np.array_equal(clustering.labels_, np.array([0, 0, 1, 1, -1, -1]))


def test_different_sample_weights():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=4)
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7],
                  [-1, -2],
                  [56, 34]])

    sample_weight = np.array([2, 2, 1, 2, 2, 2])

    # when
    clustering = dbscan.fit(X, sample_weight)

    # then (outliers label is -1)
    assert np.array_equal(clustering.labels_, np.array([0, 0, -1, -1, -1, -1]))


def test_negative_sample_weights():
    # given
    dbscan = DBScanStream(eps=1.42, min_samples=2)
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7],
                  [-1, -2],
                  [56, 34]])

    sample_weight = np.array([1, 1, -1, 3, 1, 2])

    # when
    clustering = dbscan.fit(X, sample_weight)

    # then (outliers label is -1)
    assert np.array_equal(clustering.labels_, np.array([0, 0, 1, 1, -1, 2]))


def test_two_clusters_with_outliers_manhattan_metric():
    # given
    dbscan = DBScanStream(eps=3, min_samples=2, metric='manhattan')
    X = np.array([[1, 1],
                  [2, 2],
                  [5, 6],
                  [6, 7],
                  [-1, -2],
                  [56, 34]])

    # when
    result = dbscan.fit_predict(X)

    # then
    assert np.array_equal(result, np.array([0, 0, 1, 1, -1, -1]))


def test_sample_weight_consistency():
    # given
    dbscan = DBScanStream(eps=1.2, min_samples=5)
    X = np.array([[1, 1]])
    sample_weight = np.array([1, -1, 3, 1, 2])

    # when (sample weights shape is not consistent with input data)
    with pytest.raises(AssertionError):
        dbscan.fit(X, sample_weight)


def test_invalid_metric():
    # when (DBScan is initialized with invalid metric)
    with pytest.raises(AssertionError):
        DBScanStream(eps=5, min_samples=5, metric='non existing')


if __name__ == '__main__':
    test_two_clusters_with_same_sample_weights()
