from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def kmeans_cluster(X, k=5):
    """
    Apply KMeans clustering and return labels & silhouette score
    """

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    return labels, score


def dbscan_cluster(X):
    """
    Apply DBSCAN clustering
    """

    model = DBSCAN(
        eps=0.01,
        min_samples=50
    )

    labels = model.fit_predict(X)
    return labels
