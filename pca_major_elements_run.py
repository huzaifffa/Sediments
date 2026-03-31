import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


CSV = 'Major Elements.csv'
OUT_PREFIX = 'major_elements'

if not os.path.exists(CSV):
    raise SystemExit(f"{CSV} not found in workspace")

df = pd.read_csv(CSV)

# detect sample column
if 'Samples' in df.columns:
    samples = df['Samples'].astype(str).values
    X = df.drop(columns=['Samples']).values
else:
    # fallback: first column
    samples = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].values

features = list(df.columns[1:])

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=3)
scores = pca.fit_transform(X_scaled)

scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
scores_df.insert(0, 'Sample', samples)
scores_df.to_csv(f'{OUT_PREFIX}_pca_scores.csv', index=False)

# Hierarchical clustering (Ward)
Z = linkage(X_scaled, method='ward')
# choose 3 clusters by default
clusters = fcluster(Z, t=3, criterion='maxclust')
pd.DataFrame({'Sample': samples, 'Cluster': clusters}).to_csv(f'{OUT_PREFIX}_clusters.csv', index=False)


def save_triplot():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    uniq = np.unique(clusters)
    cmap = plt.get_cmap('tab10')
    for i, u in enumerate(uniq):
        mask = clusters == u
        ax.scatter(scores[mask, 0], scores[mask, 1], scores[mask, 2], s=80, label=f'Cluster {u}', color=cmap(i))
    for i, s in enumerate(samples):
        ax.text(scores[i, 0], scores[i, 1], scores[i, 2], s, fontsize=12)
    ax.set_xlabel('PC1', fontsize=16)
    ax.set_ylabel('PC2', fontsize=16)
    ax.set_zlabel('PC3', fontsize=16)
    ax.set_title('PCA Triplot (Major Elements)', fontsize=20)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}_pca_triplot.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_dendrogram():
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=samples, leaf_rotation=90, leaf_font_size=12, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram (Ward)', fontsize=18)
    ax.set_xlabel('Sample', fontsize=14)
    ax.set_ylabel('Distance', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_heatmap():
    # sort by cluster
    order = np.argsort(clusters)
    Xs = X_scaled[order]
    names_sorted = [samples[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Xs, aspect='auto', cmap='RdYlBu_r')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Standardized value', fontsize=12)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=10)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=10)
    ax.set_title('Heatmap (Major Elements) - sorted by cluster', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print('Saving PCA triplot...')
    save_triplot()
    print('Saving dendrogram...')
    save_dendrogram()
    print('Saving heatmap...')
    save_heatmap()
    print('Done. Files:' )
    print(f" - {OUT_PREFIX}_pca_triplot.png")
    print(f" - {OUT_PREFIX}_dendrogram.png")
    print(f" - {OUT_PREFIX}_heatmap.png")
