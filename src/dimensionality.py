# src/dimensionality.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os

def apply_pca(X, n_components=3):
    """Apply PCA for dimensionality reduction"""
    print(f"Applying PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    
    print(f"✓ PCA completed")
    print(f"✓ Explained Variance Ratio: {explained_var}")
    print(f"✓ Cumulative Variance: {np.cumsum(explained_var)}")
    
    return X_pca, explained_var, pca

def apply_tsne(X, n_components=2):
    """Apply t-SNE for non-linear dimensionality reduction"""
    print(f"Applying t-SNE with {n_components} components...")
    
    tsne = TSNE(n_components=n_components, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    
    print("✓ t-SNE completed")
    return X_tsne

def get_feature_importance(pca_model, feature_names):
    """Extract feature importance from PCA components"""
    loadings = pca_model.components_[:3].T
    importance = np.abs(loadings).mean(axis=1)
    
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

def save_dimensionality_results(X_pca, explained_var, feature_importance, output_path="outputs/"):
    """Save dimensionality reduction results"""
    os.makedirs(output_path, exist_ok=True)
    
    results = {
        "pca_shape": list(X_pca.shape),
        "explained_variance": [float(x) for x in explained_var],
        "cumulative_variance": [float(x) for x in np.cumsum(explained_var)],
        "feature_importance": feature_importance
    }
    
    output_file = os.path.join(output_path, "pca_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ PCA results saved to {output_file}")
    return results