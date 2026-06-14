Anomaly Detection and Customer Segmentation Using Unsupervised Machine Learning and Deep Learning

This project presents a comprehensive machine learning pipeline for detecting anomalous financial transactions and exploring hidden patterns within highly imbalanced transaction data. Using the Credit Card Fraud Detection dataset, the project combines classical unsupervised machine learning techniques with deep learning approaches to identify fraudulent behavior, visualize complex data structures, and evaluate the effectiveness of different anomaly detection methods.

The workflow begins with extensive data preprocessing, including duplicate removal, missing value handling, feature scaling, and dataset preparation. Dimensionality reduction is then performed using Principal Component Analysis (PCA) to preserve the most significant information while reducing feature complexity. Additionally, t-SNE is utilized to visualize high-dimensional transaction data in a two-dimensional space, allowing hidden structures and potential anomaly regions to become more interpretable.

For clustering and behavioral segmentation, the project implements DBSCAN (Density-Based Spatial Clustering of Applications with Noise), a density-based clustering algorithm capable of identifying naturally occurring clusters and detecting outliers without requiring a predefined number of groups. Multiple parameter combinations are evaluated using clustering metrics such as Silhouette Score and Davies-Bouldin Index to determine the optimal clustering configuration.

The anomaly detection phase compares two fundamentally different approaches. The first approach uses One-Class Support Vector Machine (One-Class SVM), which learns the boundary of normal transaction behavior and flags observations that fall outside this boundary as anomalies. The second approach leverages a Deep Learning Autoencoder built with TensorFlow and Keras. The Autoencoder is trained exclusively on normal transactions to learn compressed latent representations of legitimate behavior. Transactions producing high reconstruction errors are identified as potential fraud cases.

To ensure a thorough evaluation, the project measures performance using Precision, Recall, F1-Score, ROC-AUC, Silhouette Score, Davies-Bouldin Index, and reconstruction error analysis. Multiple visualizations are generated throughout the workflow, including PCA projections, t-SNE embeddings, DBSCAN clustering maps, Autoencoder loss curves, reconstruction error distributions, and latent space representations.

Results demonstrate the strengths and limitations of both traditional machine learning and deep learning techniques in highly imbalanced fraud detection scenarios. The Autoencoder achieved the strongest anomaly detection performance, significantly outperforming traditional approaches in recall and ROC-AUC metrics while successfully learning meaningful latent representations of transaction behavior.

Key Technologies
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
TensorFlow
Keras
Machine Learning Techniques
Data Preprocessing & Feature Engineering
Principal Component Analysis (PCA)
t-SNE Visualization
DBSCAN Clustering
One-Class SVM Anomaly Detection
Deep Learning Autoencoder
Model Evaluation & Performance Analysis
Project Highlights
End-to-end fraud detection pipeline.
Combination of unsupervised learning and deep learning methods.
Advanced anomaly detection on highly imbalanced datasets.
Extensive visualization and dimensionality reduction techniques.
Comparative analysis between classical ML and neural network-based approaches.
Scalable framework that can be adapted to financial transactions, cybersecurity monitoring, sensor analytics, and other anomaly detection applications.

This project was developed as part of a Machine Learning course and demonstrates practical applications of modern unsupervised learning and deep learning techniques for real-world anomaly detection problems.
