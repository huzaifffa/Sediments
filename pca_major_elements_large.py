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
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

CSV = 'Major Elements.csv'
OUT_PREFIX = 'major_elements_large'
SAVE_DPI = 600

if not os.path.exists(CSV):
    raise SystemExit(f"{CSV} not found in workspace")

df = pd.read_csv(CSV)
if 'Samples' in df.columns:
    samples = df['Samples'].astype(str).values
    X = df.drop(columns=['Samples']).values
else:
    samples = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].values

features = list(df.columns[1:])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
scores = pca.fit_transform(X_scaled)

Z = linkage(X_scaled, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')


def save_triplot():
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    uniq = np.unique(clusters)
    cmap = plt.get_cmap('tab10')
    for i, u in enumerate(uniq):
        mask = clusters == u
        ax.scatter(scores[mask, 0], scores[mask, 1], scores[mask, 2], s=160, label=f'Cluster {u}', color=cmap(i), edgecolor='k')
    for i, s in enumerate(samples):
        ax.text(scores[i, 0], scores[i, 1], scores[i, 2], s, fontsize=14)
    ax.set_xlabel('PC1', fontsize=18)
    ax.set_ylabel('PC2', fontsize=18)
    ax.set_zlabel('PC3', fontsize=18)
    ax.set_title('PCA Triplot (Major Elements) - Large', fontsize=22)
    ax.legend()
    plt.tight_layout()
    png = f'{OUT_PREFIX}_pca_triplot.png'
    pdf = f'{OUT_PREFIX}_pca_triplot.pdf'
    plt.savefig(png, dpi=SAVE_DPI, bbox_inches='tight')
    plt.savefig(pdf, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()


def save_dendrogram():
    fig, ax = plt.subplots(figsize=(20, 12))
    dendrogram(Z, labels=samples, leaf_rotation=90, leaf_font_size=18, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram (Ward) - Large', fontsize=22)
    ax.set_xlabel('Sample', fontsize=18)
    ax.set_ylabel('Distance', fontsize=18)
    plt.tight_layout()
    png = f'{OUT_PREFIX}_dendrogram.png'
    pdf = f'{OUT_PREFIX}_dendrogram.pdf'
    plt.savefig(png, dpi=SAVE_DPI, bbox_inches='tight')
    plt.savefig(pdf, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()


def save_heatmap():
    order = np.argsort(clusters)
    Xs = X_scaled[order]
    names_sorted = [samples[i] for i in order]
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(Xs, aspect='auto', cmap='RdYlBu_r')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Standardized value', fontsize=16)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=12)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=14)
    ax.set_title('Heatmap (Major Elements) - Large', fontsize=22)
    plt.tight_layout()
    png = f'{OUT_PREFIX}_heatmap.png'
    pdf = f'{OUT_PREFIX}_heatmap.pdf'
    plt.savefig(png, dpi=SAVE_DPI, bbox_inches='tight')
    plt.savefig(pdf, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print('Creating large, high-resolution images...')
    save_triplot()
    print('Saved triplot')
    save_dendrogram()
    print('Saved dendrogram')
    save_heatmap()
    print('Saved heatmap')
    print('Done. Files:')
    print(f' - {OUT_PREFIX}_pca_triplot.png/pdf')
    print(f' - {OUT_PREFIX}_dendrogram.png/pdf')
    print(f' - {OUT_PREFIX}_heatmap.png/pdf')
