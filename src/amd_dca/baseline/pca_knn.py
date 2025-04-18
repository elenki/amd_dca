from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


def pca_knn_denoise(
    X_train_log: np.ndarray,
    X_target_log: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Performs PCA followed by KNN regression denoising on log1p counts.

    Args:
        X_train_log: (n_train, n_genes) log1p(counts) training data
        X_target_log: (n_target, n_genes) log1p(counts) to denoise
        n_components: number of PCA components
        n_neighbors: number of neighbors in KNN
        random_state: seed for PCA

    Returns:
        X_target_denoised_log: (n_target, n_genes) denoised in log space
    """
    # 1) PCA reduction
    pca = PCA(n_components=n_components, random_state=random_state)
    Z_train = pca.fit_transform(X_train_log)
    Z_target = pca.transform(X_target_log)

    # 2) KNN regression
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(Z_train, X_train_log)
    X_target_denoised_log = knn.predict(Z_target)
    return X_target_denoised_log
