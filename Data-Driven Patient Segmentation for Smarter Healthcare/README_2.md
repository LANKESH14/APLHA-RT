
# Clustering and PCA Analysis of Hospital Data

This repository contains a project focused on clustering patient data from a synthetic hospital dataset. The aim is to group patients based on their medical history and treatment adherence, utilizing various preprocessing techniques, Principal Component Analysis (PCA), and K-means clustering.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Steps](#steps)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Correlation Analysis](#2-correlation-analysis)
  - [3. PCA (Principal Component Analysis)](#3-pca-principal-component-analysis)
  - [4. Clustering using K-means](#4-clustering-using-k-means)
  - [5. Optimal Number of Clusters](#5-optimal-number-of-clusters)
- [Validation](#validation)
- [Results](#results)

## Dataset

The dataset used in this project is a synthetic hospital data file, `synthetic_hospital_data_updated.csv`. It contains various features like age, gender, medical history, and treatment adherence score, which are used to cluster patients.

## Requirements

To run this project, you'll need the following libraries:

```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
```

You can install these dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Steps

### 1. Data Preprocessing

The first step involves cleaning the data by dropping irrelevant columns (e.g., `Patient ID`). We also preprocess numeric and categorical features using `StandardScaler` and `OneHotEncoder` respectively, to prepare the data for clustering.

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define preprocessing for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Age', 'Number of Visits', 'Total Medical Expenses', 'Treatment Adherence Score']),
        ('cat', categorical_transformer, ['Gender', 'Medical History'])
    ])
X_preprocessed = preprocessor.fit_transform(data)
```

### 2. Correlation Analysis

We visualize the correlation between features using a heatmap to understand how different variables relate to each other.

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
```

### 3. PCA (Principal Component Analysis)

To reduce the dimensionality of the data, PCA is applied. This helps in visualizing clusters more clearly and reduces noise in the clustering process.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_preprocessed)
```

### 4. Clustering using K-means

We use the K-means clustering algorithm to group patients based on their features. The algorithm is applied to the PCA-reduced data, and different numbers of clusters are tested to find the optimal number.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)
```

### 5. Optimal Number of Clusters

We determine the optimal number of clusters using the Elbow Method and the Silhouette Score. This helps us choose the best number of clusters that fit the data.

```python
import matplotlib.pyplot as plt

# Elbow Method and Silhouette Score
sse = []
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Plot the results
plt.plot(cluster_range, sse, label='SSE')
plt.plot(cluster_range, silhouette_scores, label='Silhouette Score')
plt.legend()
plt.show()
```

## Validation

After clustering, we evaluated the results using three key metrics:

- **Silhouette Score**: `0.185`  
  - A low score indicates that the clusters are not well-separated, suggesting potential overlap between clusters or suboptimal clustering.

- **Calinski-Harabasz Index**: `1685.58`  
  - This index measures the ratio of the sum of between-cluster dispersion to within-cluster dispersion. A higher score indicates better-defined clusters. While the score is moderate, there may still be room for improvement.

- **Davies-Bouldin Index**: `1.297`  
  - This index evaluates the average similarity ratio of each cluster to its most similar cluster. Lower values signify better cluster separation, and a score of 1.297 suggests moderate cluster distinctiveness, but with opportunities for refinement.

## Results

The final model identified 8 clusters of patients, with each cluster representing a group with similar medical profiles. The PCA analysis helped reduce the dataset's complexity, and the K-means algorithm successfully categorized the patients into meaningful groups.
