# src/train.py
import os
import sys
import mlflow
import mlflow.sklearn
import json
import logging
from datetime import datetime
import gc

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from data_loader import load_data
from preprocessing import clean_data
from features import select_features
from clustering import kmeans_cluster, dbscan_cluster
from dimensionality import apply_pca, get_feature_importance, save_dimensionality_results

# =================== LOGGING SETUP ====================
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== MLflow SETUP ====================
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
mlflow.set_experiment("Chicago Crime Clustering")

logger.info("="*80)
logger.info("STARTING CHICAGO CRIME CLUSTERING PIPELINE")
logger.info("="*80)

try:
    # -------- STEP 1: Load Data --------
    logger.info("STEP 1: Loading Chicago crime data...")
    df = load_data("data/raw/crimes.csv")
    logger.info(f"✓ Original dataset shape: {df.shape}")
    
    # -------- STEP 2: Clean Data --------
    logger.info("STEP 2: Cleaning and preprocessing data...")
    df = clean_data(df)
    logger.info(f"✓ After cleaning shape: {df.shape}")
    
    # ✨ SAVE PROCESSED DATA 
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/crime_cleaned.csv", index=False)
    logger.info("✓ Cleaned data saved to data/processed/crime_cleaned.csv")
    del df
    gc.collect()
    # -------- STEP 3: Sample Data --------
    logger.info("STEP 3: Sampling 50,000 records for processing...")
    df = load_data("data/processed/crime_cleaned.csv")
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
    logger.info(f"✓ Sampled dataset shape: {df.shape}")
    
    # -------- STEP 4: Feature Selection --------
    logger.info("STEP 4: Selecting features for clustering...")
    X = select_features(df)
    logger.info(f"✓ Features selected: {X.columns.tolist()}")
    logger.info(f"✓ Feature matrix shape: {X.shape}")
    
    # -------- STEP 5: Feature Scaling --------
    logger.info("STEP 5: Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("✓ Features scaled successfully")
    
    # -------- STEP 6: Dimensionality Reduction (PCA) --------
    logger.info("STEP 6: Applying PCA for feature reduction...")
    X_pca, explained_var, pca_model = apply_pca(X_scaled, n_components=3)
    
    feature_importance = get_feature_importance(pca_model, X.columns.tolist())
    logger.info(f"✓ PCA completed. Explained variance: {np.cumsum(explained_var)[-1]:.4f}")
    logger.info(f"✓ Top 5 Important Features: {list(feature_importance.items())[:5]}")
    
    # Save PCA results
    save_dimensionality_results(X_pca, explained_var, feature_importance)
    
    # -------- STEP 7: K-Means Clustering --------
    logger.info("STEP 7: Training K-Means clustering models...")
    #For clustering we only need coordinates
    df = df[['Latitude','Longitude']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    kmeans_results = []
    best_kmeans_k = None
    best_kmeans_score = -1
    
    for k in range(3, 11):
        logger.info(f"  Testing K-Means with K={k}...")
        
        with mlflow.start_run(nested=True):
            labels, score = kmeans_cluster(X_scaled, k=k)
            
            mlflow.log_param("algorithm", "kmeans")
            mlflow.log_param("clusters", k)
            mlflow.log_metric("silhouette_score", score)
            
            # Calculate Davies-Bouldin Index
            db_score = davies_bouldin_score(X_scaled, labels)
            mlflow.log_metric("davies_bouldin_score", db_score)
            
            logger.info(f"  ✓ K={k}: Silhouette={score:.4f}, Davies-Bouldin={db_score:.4f}")
            
            kmeans_results.append({
                "k": k,
                "silhouette_score": float(score),
                "davies_bouldin_score": float(db_score)
            })
            
            if score > best_kmeans_score:
                best_kmeans_score = score
                best_kmeans_k = k
    
    logger.info(f"✓ Best K-Means: K={best_kmeans_k}, Score={best_kmeans_score:.4f}")
    
    # -------- STEP 8: DBSCAN Clustering --------
    logger.info("STEP 8: Training DBSCAN clustering...")
    
    with mlflow.start_run(nested=True):
        dbscan_labels = dbscan_cluster(X_scaled)
        
        # Filter out noise points (-1 label) for silhouette calculation
        mask = dbscan_labels != -1
        if np.sum(mask) > 1:
            db_score_dbscan = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        else:
            db_score_dbscan = -1
        
        mlflow.log_param("algorithm", "dbscan")
        mlflow.log_param("eps", 0.01)
        mlflow.log_param("min_samples", 50)
        mlflow.log_metric("silhouette_score", db_score_dbscan)
        mlflow.log_metric("n_clusters", len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
        
        logger.info(f"✓ DBSCAN: Silhouette={db_score_dbscan:.4f}")
    
    # -------- STEP 9: Hierarchical Clustering --------
    logger.info("STEP 9: Training Hierarchical clustering...")
    
    from sklearn.cluster import AgglomerativeClustering
    gc.collect()
    with mlflow.start_run(nested=True):
        hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
        hier_labels = hierarchical.fit_predict(X_scaled)
        hier_score = silhouette_score(X_scaled, hier_labels)
        
        mlflow.log_param("algorithm", "hierarchical")
        mlflow.log_param("linkage", "ward")
        mlflow.log_metric("silhouette_score", hier_score)
        
        logger.info(f"✓ Hierarchical: Silhouette={hier_score:.4f}")
    
    # -------- STEP 10: Save Results --------
    logger.info("STEP 10: Saving clustering results...")
    
    os.makedirs("outputs", exist_ok=True)
    
    results = {
        "dataset_info": {
            "original_shape":str(df.shape),
            "features": X.columns.tolist()
        },
        "kmeans_results": kmeans_results,
        "best_kmeans": {
            "k": int(best_kmeans_k),
            "silhouette_score": float(best_kmeans_score)
        },
        "dbscan_results": {
            "silhouette_score": float(db_score_dbscan)
        },
        "hierarchical_results": {
            "silhouette_score": float(hier_score)
        },
        "feature_importance": feature_importance
    }
    
    with open("outputs/clustering_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("✓ Results saved to outputs/clustering_results.json")
    
    # -------- STEP 11: Register Model --------
    logger.info("STEP 11: Registering best model in MLflow...")
    
    with mlflow.start_run(run_name="best_kmeans_model"):
        labels, score = kmeans_cluster(X_scaled, k=best_kmeans_k)
        
        mlflow.log_param("algorithm", "kmeans")
        mlflow.log_param("clusters", best_kmeans_k)
        mlflow.log_metric("silhouette_score", score)
        mlflow.log_artifact("outputs/clustering_results.json")
        
        logger.info("✓ Best model registered in MLflow")
    
    logger.info("="*80)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("")
    logger.info("📊 FILES CREATED:")
    logger.info("  ✓ data/processed/crime_cleaned.csv")
    logger.info("  ✓ outputs/clustering_results.json")
    logger.info("  ✓ outputs/pca_results.json")
    logger.info("  ✓ logs/training_*.log")
    logger.info("")
    logger.info("📊 NEXT STEPS:")
    logger.info("1. View MLflow experiments:")
    logger.info("   mlflow ui")
    logger.info("")
    logger.info("2. Start Streamlit dashboard (in another terminal):")
    logger.info("   streamlit run app/Home.py")
    logger.info("")
    logger.info("="*80)
    
except Exception as e:
    logger.error(f"❌ ERROR in training pipeline: {str(e)}", exc_info=True)
    raise