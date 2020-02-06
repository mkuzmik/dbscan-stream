# Streamable DBSCAN

# Overview

This library implements DBSCAN clustering algorithm. 
It is compatible with SKLEARN API and may be learned iteratively.

# Requirements

- Python 3.6 or later
- virtualenv tool

# How to run

Create virtual environment:
`virtualenv venv`

Install dependencies: 
`make install-deps`

Run tests: 
`make test`

# API

### Create DBScanStream object

```python
dbscan = DBScanStream(eps=1.5, min_samples=5, metric='manhattan', leaf_size=50)
```

where:
- `eps` is maximum distance for a point to be considered neighbour 
- `min_samples` is minimal amount of neighbours to form a cluster
- `metric` (_optional_) is indicating how the distance will be calculated, can be one of `‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’`, by default it's `euclidean`
- `leaf_size` (_optional_) it is passed to the search tree, it's affecting tree build and distance query speed, by default it's `30`

### `fit`

It computes neighbourhood from scratch, basing on provided samples and performs clustering. 

```python
X = np.array([[1, 1],[2, 2],[5, 6],[6, 7],[-1, -2],[56, 34]])
weights = [1, 1.2, 1.2, 1.3, 0.2, 0.9, 1]
new_dbscan = dbscan.fit(X, weights)
```

where:
- `X` training points to the cluster, should be array like shape, with arrays/tuples with consistent size
- `weights` (_optional_) weight of each sample, should have the same amount of records as `X`, by default every record has weight of `1`

It returns DBScanStream object with proper cluster labels available under `labels_` attribute.
Label equal to `-1` indicates that point is an outlier. 

### `fit_predict`

It does the same as `fit` but it returns cluster labels;

```python
X = np.array([[1, 1],[2, 2],[5, 6],[6, 7],[-1, -2],[56, 34]])
weights = [1, 1.2, 1.2, 1.3, 0.2, 0.9, 1]
labels = dbscan.fit_predict(X, weights)
```

### `partial_fit` 

It updates existing neighbourhood (if it's run for a first time it works as `fit` method) with provided samples and does clustering on updated data.
It accepts the same arguments as `fit` and `fit_predict`.
It returns updated DBScanStream object.

```python
X = np.array([[1, 1],[2, 2],[5, 6],[6, 7],[-1, -2],[56, 34]])
weights = [1, 1.2, 1.2, 1.3, 0.2, 0.9, 1]
dbscan.fit(X, weights)

new_X1 = np.array([[1,2], [6,7]])
new_X2 = np.array([[9,9]])

dbscan.partial_fit(new_X1)
dbscan.partial_fit(new_X2)
```