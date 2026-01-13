import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Load the data
print("Loading metal concentration data...")
df = pd.read_csv('Metal concentration.csv')

# Display basic information
print(f"\nDataset shape: {df.shape}")
print(f"Number of samples: {len(df)}")
print(f"\nFirst few rows:")
print(df.head())

# Prepare data for clustering
# Remove the 'Points' column (sample names) and use it as index
sample_names = df['Points'].values
X = df.drop('Points', axis=1).values

# Remove rows with all NaN values (empty rows at the end)
valid_indices = ~np.all(np.isnan(X), axis=1)
X = X[valid_indices]
sample_names = sample_names[valid_indices]

print(f"\nValid samples for analysis: {len(sample_names)}")

# Handle missing values by removing rows with any NaN
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
sample_names = sample_names[mask]

print(f"Samples after removing NaN values: {len(sample_names)}")

# Standardize the features (important for clustering)
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform agglomerative hierarchical clustering with different linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']

print("\nPerforming hierarchical clustering with different linkage methods...\n")

# Create a figure with multiple dendrograms
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    # Compute linkage matrix
    Z = linkage(X_scaled, method=method)
    
    # Plot dendrogram
    ax = axes[idx]
    dendrogram(Z, labels=sample_names, ax=ax, leaf_font_size=8)
    ax.set_title(f'Hierarchical Clustering Dendrogram ({method.upper()} Linkage)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig('dendrograms_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dendrograms_comparison.png")
plt.show()

# Use Ward linkage (most common choice)
print("\n" + "="*60)
print("WARD LINKAGE ANALYSIS (Primary Method)")
print("="*60)

Z_ward = linkage(X_scaled, method='ward')

# Create a larger, detailed dendrogram with Ward linkage
fig, ax = plt.subplots(figsize=(20, 10))
dendro = dendrogram(Z_ward, labels=sample_names, ax=ax, 
                    leaf_font_size=10, color_threshold=0)
ax.set_title('Agglomerative Hierarchical Clustering - Ward Linkage', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Sample', fontsize=13, fontweight='bold')
ax.set_ylabel('Euclidean Distance', fontsize=13, fontweight='bold')
ax.axhline(y=ax.get_ylim()[1] * 0.5, c='red', linestyle='--', linewidth=2, label='Suggested Cutoff')
ax.legend(fontsize=11)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('dendrogram_ward_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dendrogram_ward_detailed.png")
plt.show()

# Determine optimal number of clusters using elbow method on last distances
print("\nLast 20 merges (distances):")
print("-" * 40)
last_merges = Z_ward[-20:, 2]
for i, distance in enumerate(last_merges, start=len(Z_ward)-19):
    print(f"Merge {i}: Distance = {distance:.4f}")

# Cut the dendrogram to form clusters
# Try different numbers of clusters
n_clusters_list = [3, 4, 5, 6, 7]

print(f"\n{'Number of Clusters':<20} {'Cluster Distribution'}")
print("-" * 60)

for n_clusters in n_clusters_list:
    clusters = fcluster(Z_ward, n_clusters, criterion='maxclust')
    unique, counts = np.unique(clusters, return_counts=True)
    distribution = ', '.join([f"Cluster {u}: {c} samples" for u, c in zip(unique, counts)])
    print(f"{n_clusters:<20} {distribution}")

# Use 4 clusters as an example (you can adjust based on your needs)
optimal_clusters = 4
clusters = fcluster(Z_ward, optimal_clusters, criterion='maxclust')

print(f"\n{'='*60}")
print(f"Clustering Results (k={optimal_clusters} clusters)")
print(f"{'='*60}")

# Create a DataFrame with cluster assignments
result_df = pd.DataFrame({
    'Sample': sample_names,
    'Cluster': clusters
})

print("\nCluster Assignments:")
print(result_df.sort_values('Cluster'))

# Save results
result_df.to_csv('cluster_assignments.csv', index=False)
print("\n✓ Saved: cluster_assignments.csv")

# Visualize clusters with a heatmap
fig, ax = plt.subplots(figsize=(14, 10))

# Sort data by cluster for better visualization
sorted_indices = np.argsort(clusters)
X_sorted = X_scaled[sorted_indices]
clusters_sorted = clusters[sorted_indices]
names_sorted = sample_names[sorted_indices]

# Create heatmap
sns.heatmap(X_sorted, cmap='RdYlBu_r', xticklabels=df.columns[1:], 
            yticklabels=names_sorted, cbar_kws={'label': 'Standardized Concentration'},
            ax=ax)
ax.set_title(f'Metal Concentrations Heatmap (Sorted by {optimal_clusters} Clusters)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Metal Elements', fontsize=12, fontweight='bold')
ax.set_ylabel('Samples', fontsize=12, fontweight='bold')

# Add cluster boundaries
cluster_boundaries = np.where(np.diff(clusters_sorted) != 0)[0] + 1
for boundary in cluster_boundaries:
    ax.axhline(y=boundary, color='black', linewidth=2)

plt.tight_layout()
plt.savefig('heatmap_clusters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: heatmap_clusters.png")
plt.show()

# Summary statistics for each cluster
print(f"\n{'='*60}")
print("CLUSTER SUMMARY STATISTICS")
print(f"{'='*60}")

for cluster_id in np.unique(clusters):
    cluster_mask = clusters == cluster_id
    cluster_samples = sample_names[cluster_mask]
    print(f"\nCluster {cluster_id} ({np.sum(cluster_mask)} samples):")
    print(f"Samples: {', '.join(cluster_samples)}")
    print(f"Mean concentrations (original scale):")
    
    cluster_data = df[df['Points'].isin(cluster_samples)].drop('Points', axis=1)
    means = cluster_data.mean()
    print(means)

# Create a distance matrix heatmap
print("\nGenerating distance matrix heatmap...")
dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(dist_matrix, xticklabels=sample_names, yticklabels=sample_names,
            cmap='YlOrRd', square=True, cbar_kws={'label': 'Euclidean Distance'},
            ax=ax)
ax.set_title('Euclidean Distance Matrix Between Samples', 
             fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('distance_matrix_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: distance_matrix_heatmap.png")
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("1. dendrograms_comparison.png - Comparison of different linkage methods")
print("2. dendrogram_ward_detailed.png - Detailed Ward linkage dendrogram")
print("3. cluster_assignments.csv - Sample-to-cluster assignments")
print("4. heatmap_clusters.png - Heatmap of standardized metal concentrations")
print("5. distance_matrix_heatmap.png - Distance matrix between all samples")
