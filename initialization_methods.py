import numpy as np

def forgy(X, row_count, n_clusters):
    return X [ np.random.choice(row_count, size=n_clusters, replace=False) ]

def macqueen(X, n_clusters):
    return X [:n_clusters]

def maximin(X, n_clusters):
    X_ = np.copy(X)
    initial_centers = np.zeros((n_clusters, X_.shape[1]))
    X_norms = np.linalg.norm(X_, axis = 1)
    X_norms_max_i = X_norms.argmax()
    initial_centers[0] = X_[X_norms_max_i]
    X_ = np.delete(X_, X_norms_max_i, axis = 0)
    for i in range(1, n_clusters):
        distances = np.zeros((X_.shape[0], i))
        for index, center in enumerate(initial_centers[:i]):
            distances[:, index] = np.linalg.norm(X_ - center, axis = 1)

        max_min_index = distances.min(axis = 1).argmax()

        initial_centers[i] = X_[max_min_index]
        X_ = np.delete(X_, max_min_index, axis = 0)
        
    return initial_centers
        
        