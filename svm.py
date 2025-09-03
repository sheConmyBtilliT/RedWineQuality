from sklearn.svm import SVC


def get_svm_model(kernel='rbf'):
    """Returns a Support Vector Machine classifier."""
    return SVC(kernel=kernel, probability=True, random_state=42)
