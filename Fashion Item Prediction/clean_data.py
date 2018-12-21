from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
import numpy as np

def num_pipeline (X):
    """
    Perform normalization and other pre-steps.
    """  

    # This scales the features to have a mean of zero and unit variance
    pipeline = Pipeline([
        ('std scaler', StandardScaler()),
        ])
    X_scaled = pipeline.fit_transform(X)
    return (X_scaled)

def inc_pca (X):
    """
    Perform incremental PCA to reduce dimensions while keeping high variance.
    """
    
    n_batches = 100
    inc_pca = IncrementalPCA (n_components = 14*14)
    for X_batch in np.array_split (X, n_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X)
    return (X_reduced)