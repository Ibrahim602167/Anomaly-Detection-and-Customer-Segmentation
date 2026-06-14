# =========================
#  Import Libraries
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# =========================
# Random Seeds
# =========================

np.random.seed(42)
tf.random.set_seed(42)

# =========================
#  Load Dataset
# =========================

script_dir = os.path.dirname(os.path.abspath(__file__))

zip_path = os.path.join(
    script_dir,
    "creditcard.csv.zip"
)

# fallback path
if not os.path.exists(zip_path):

    zip_path = r"C:\Users\user\OneDrive\Desktop\Project-ML\creditcard.csv.zip"

if not os.path.exists(zip_path):

    raise FileNotFoundError(
        f"Dataset not found.\n"
        f"Please place 'creditcard.csv.zip' next to this script:\n"
        f"{script_dir}"
    )

# [FIX] Reduce rows to avoid MemoryError
df = pd.read_csv(
    zip_path,
    compression="zip",
    nrows=20000
)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ---------------------------------------------
#A. Data preprocessing and feature engineering 
# ---------------------------------------------


# =================
# 1. Data Checking
# =================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicated Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

print("\nShape After Removing Duplicates:")
print(df.shape)

# =========================
# 2. Prepare Features and Labels
# =========================

if "Class" in df.columns:

    y_true = df["Class"].reset_index(drop=True)

    X = df.drop(
        "Class",
        axis=1
    )

else:

    y_true = None
    X = df.copy()

X = X.select_dtypes(include=[np.number])

X = X.fillna(X.mean())

X = X.reset_index(drop=True)

print("\nFeatures Shape:")
print(X.shape)

if y_true is not None:

    print("\nClass Distribution:")
    print(y_true.value_counts())

# =========================
# 3. Feature Scaling
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData Scaling Completed.")

#-------------------------
#Dimensionality reduction
#-------------------------

# ==========
# 1. PCA
# ==========

pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    s=5
)

plt.title("PCA 2D Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")



# Explained variance

pca_full = PCA()

pca_full.fit(X_scaled)

plt.figure(figsize=(8, 6))

plt.plot(
    np.cumsum(
        pca_full.explained_variance_ratio_
    )
)

plt.title("Cumulative Explained Variance by PCA")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

plt.grid()



# PCA 95%

pca_95 = PCA(n_components=0.95)

X_pca_95 = pca_95.fit_transform(X_scaled)

print("\nOriginal Features:", X_scaled.shape[1])

print("Reduced Features:", X_pca_95.shape[1])

# =========================
#  t-SNE Visualization
# =========================

sample_size = min(
    5000,
    X_scaled.shape[0]
)

sample_indices = np.random.choice(
    X_scaled.shape[0],
    sample_size,
    replace=False#بدون تكرار
)

X_sample = X_scaled[sample_indices]

if y_true is not None:

    y_sample = y_true.iloc[sample_indices]

else:

    y_sample = None

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    learning_rate="auto",
    init="pca"
)

X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(8, 6))

if y_sample is not None:

    plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_sample,
        s=5,
        cmap="coolwarm"
    )

    plt.colorbar(label="Class")

else:

    plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        s=5
    )

plt.title("t-SNE 2D Visualization")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

#-----------------------
#Clustering techniques
#-----------------------

# ========================================
# 1. DBSCAN Clustering + Parameter Tuning
# ========================================

print("\nRunning DBSCAN Parameter Tuning...")

eps_values = [0.5, 1, 2, 3, 5]
min_samples_values = [5, 10, 15]

best_score = -1
best_eps = None
best_min_samples = None
best_labels = None
best_num_clusters = None
best_num_noise = None

# Use a sample for faster and safer metric calculation
metric_sample_size = min(
    5000,
    X_pca_2d.shape[0]
)

metric_idx = np.random.choice(
    X_pca_2d.shape[0],
    metric_sample_size,
    replace=False
)

for eps in eps_values:

    for min_samples in min_samples_values:

        dbscan_test = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = dbscan_test.fit_predict(X_pca_2d)

        current_num_clusters = len(set(labels)) - (
            1 if -1 in labels else 0
        )

        current_num_noise = np.sum(labels == -1)

        # Silhouette Score needs at least 2 clusters
        if current_num_clusters >= 2:

            try:

                current_score = silhouette_score(
                    X_pca_2d[metric_idx],
                    labels[metric_idx]
                )

                print(
                    f"eps={eps}, "
                    f"min_samples={min_samples}, "
                    f"clusters={current_num_clusters}, "
                    f"noise={current_num_noise}, "
                    f"silhouette={current_score:.4f}"
                )

                if current_score > best_score:

                    best_score = current_score
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
                    best_num_clusters = current_num_clusters
                    best_num_noise = current_num_noise

            except Exception as e:

                print(
                    f"Skipped eps={eps}, "
                    f"min_samples={min_samples} "
                    f"because metric calculation failed."
                )

        else:

            print(
                f"Skipped eps={eps}, "
                f"min_samples={min_samples}: "
                f"not enough clusters."
            )

print("\n========== Best DBSCAN Parameters ==========")
print("Best eps:", best_eps)
print("Best min_samples:", best_min_samples)
print("Best Silhouette Score:", best_score)

# Final DBSCAN model using the best parameters

dbscan = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples
)

dbscan_labels = dbscan.fit_predict(X_pca_2d)

print("\nDBSCAN Labels:")
print(np.unique(dbscan_labels))

num_clusters = len(set(dbscan_labels)) - (
    1 if -1 in dbscan_labels else 0
)

num_noise = np.sum(dbscan_labels == -1)

print("\nNumber of Clusters:", num_clusters)

print("Noise Points:", num_noise)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=dbscan_labels,
    s=5,
    cmap="tab10"
)

plt.title(
    f"DBSCAN Clustering "
    f"(eps={best_eps}, min_samples={best_min_samples})"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(label="Cluster")



print("\nCluster Distribution:")
print(pd.Series(dbscan_labels).value_counts())

# Final metrics using the same sampled metric approach

if num_clusters >= 2:
#Silhouette Score
    sil_score = silhouette_score(
        X_pca_2d[metric_idx],
        dbscan_labels[metric_idx]
    )
#Davies-Bouldin Index
    db_score = davies_bouldin_score(
        X_pca_2d[metric_idx],
        dbscan_labels[metric_idx]
    )

    print("\nSilhouette Score:", sil_score)

    print("Davies-Bouldin Index:", db_score)

else:

    print("\nNot enough clusters for evaluation.")

#------------------
#Anomaly detection 
#------------------

# =========================
# 1. One-Class SVM
# =========================

print("\nTraining One-Class SVM...",
      "\n-----------------")

svm_sample_size = min(
    5000,
    X_pca_95.shape[0]
)

svm_sample_idx = np.random.choice(
    X_pca_95.shape[0],
    svm_sample_size,
    replace=False
)

ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.01
)

ocsvm.fit(
    X_pca_95[svm_sample_idx]
)

svm_pred = ocsvm.predict(X_pca_95)

svm_anomaly = np.where(
    svm_pred == -1,
    1,
    0
)

print("-----------------"
   , "\nDetected Anomalies:",
    np.sum(svm_anomaly)
)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=svm_anomaly,
    s=5,
    cmap="coolwarm"
)

plt.title("One-Class SVM")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(label="Anomaly")



if y_true is not None:

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            svm_anomaly
        )
    )

    print(
        "Precision:",
        precision_score(
            y_true,
            svm_anomaly
        )
    )

    print(
        "Recall:",
        recall_score(
            y_true,
            svm_anomaly
        )
    )

    print(
        "F1-score:",
        f1_score(
            y_true,
            svm_anomaly
        )
    )

    try:

        print(
            "ROC-AUC:",
            roc_auc_score(
                y_true,
                svm_anomaly
            )
        )

    except Exception:

        print("ROC-AUC failed.")

#-----------------------------------------
#Deep learning using TensorFlow and Keras
#-----------------------------------------

# =========================
# 1. Autoencoder
# =========================

input_dim = X_scaled.shape[1]

if y_true is not None:

    X_train_ae = X_scaled[
        y_true.values == 0
    ]

    print(
        f"\nTraining Autoencoder on "
        f"{X_train_ae.shape[0]} "
        f"normal transactions."
    )

else:

    X_train_ae = X_scaled

input_layer = Input(
    shape=(input_dim,)
)

encoded = Dense(
    32,
    activation="relu"
)(input_layer)

encoded = Dense(
    16,
    activation="relu"
)(encoded)

latent = Dense(
    8,
    activation="relu",
    name="latent_space"
)(encoded)

decoded = Dense(
    16,
    activation="relu"
)(latent)

decoded = Dense(
    32,
    activation="relu"
)(decoded)

output_layer = Dense(
    input_dim,
    activation="linear"
)(decoded)

autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train_ae,
    X_train_ae,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    shuffle=True,
    verbose=1
)

# =========================
# 2. Loss Curve
# =========================

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Autoencoder Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid()



# =========================
# 3. Reconstruction Error
# =========================

X_pred = autoencoder.predict(X_scaled)

reconstruction_error = np.mean(
    np.square(X_scaled - X_pred),
    axis=1
)

plt.figure(figsize=(8, 6))

plt.hist(
    reconstruction_error,
    bins=50
)

plt.title("Reconstruction Error")

plt.xlabel("Error")
plt.ylabel("Frequency")



if y_true is not None:

    normal_errors = reconstruction_error[
        y_true.values == 0
    ]

    threshold = np.percentile(
        normal_errors,
        95
    )

else:

    threshold = np.percentile(
        reconstruction_error,
        99
    )

autoencoder_anomaly = np.where(
    reconstruction_error > threshold,
    1,
    0
)

print("\nThreshold:", threshold)

print(
    "Detected Autoencoder Anomalies:",
    np.sum(autoencoder_anomaly)
)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=autoencoder_anomaly,
    s=5,
    cmap="coolwarm"
)

plt.title("Autoencoder Anomaly Detection")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(label="Anomaly")



if y_true is not None:

    print("\nAutoencoder Report:")

    print(
        classification_report(
            y_true,
            autoencoder_anomaly
        )
    )

    print(
        "Precision:",
        precision_score(
            y_true,
            autoencoder_anomaly
        )
    )

    print(
        "Recall:",
        recall_score(
            y_true,
            autoencoder_anomaly
        )
    )

    print(
        "F1-score:",
        f1_score(
            y_true,
            autoencoder_anomaly
        )
    )

    try:

        print(
            "ROC-AUC:",
            roc_auc_score(
                y_true,
                reconstruction_error
            )
        )

    except Exception:

        print("ROC-AUC failed.")

# =========================
# 4. Latent Representation
# =========================

encoder = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.get_layer(
        "latent_space"
    ).output
)

X_latent = encoder.predict(X_scaled)

latent_pca = PCA(
    n_components=2
)

X_latent_2d = latent_pca.fit_transform(
    X_latent
)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_latent_2d[:, 0],
    X_latent_2d[:, 1],
    c=autoencoder_anomaly,
    s=5,
    cmap="coolwarm"
)

plt.title("Latent Representation")

plt.xlabel("Latent PC1")
plt.ylabel("Latent PC2")

plt.colorbar(label="Anomaly")



# =========================
#  Comparison Table
# =========================

results = [

    {
        "Method": "DBSCAN",
        "Purpose": "Clustering",
        "Detected": int(
            np.sum(dbscan_labels == -1)
        ),
        "Notes": "Density-based"
    },

    {
        "Method": "One-Class SVM",
        "Purpose": "Anomaly Detection",
        "Detected": int(
            np.sum(svm_anomaly)
        ),
        "Notes": "Traditional ML"
    },

    {
        "Method": "Autoencoder",
        "Purpose": "Deep Learning",
        "Detected": int(
            np.sum(autoencoder_anomaly)
        ),
        "Notes":
        "Trained on normal data"
    }
]

comparison_df = pd.DataFrame(results)

print("\nComparison Table:")
print(comparison_df)

# =========================
#  Final Summary
# =========================

print("\n================ SUMMARY ================")

print("Dataset Records:", df.shape[0])

print("Dataset Features:", X.shape[1])

print("PCA Features:", X_pca_95.shape[1])

print("DBSCAN Clusters:", num_clusters)

print("DBSCAN Noise:", num_noise)

print(
    "One-Class SVM Anomalies:",
    np.sum(svm_anomaly)
)

print(
    "Autoencoder Anomalies:",
    np.sum(autoencoder_anomaly)
)



print("========================================")
