from sklearn.neighbors import KNeighborsClassifier


def get_knn_model(n_neighbors=5):
    """Returns a k-Nearest Neighbors classifier."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)
