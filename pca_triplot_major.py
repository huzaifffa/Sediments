import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

SCORES_CSV = 'pca_scores_Major_elements.csv'
CLUSTER_CSV = 'cluster_assignments_Major_elements.csv'
OUT_PNG = 'pca_triplot_major_elements.png'

if not os.path.exists(SCORES_CSV):
    raise SystemExit(f"{SCORES_CSV} not found")
df = pd.read_csv(SCORES_CSV)
if os.path.exists(CLUSTER_CSV):
    clusters = pd.read_csv(CLUSTER_CSV)
    df = df.merge(clusters, on='Sample', how='left')

required = {'Sample', 'PC1', 'PC2', 'PC3'}
if not required.issubset(df.columns):
    raise SystemExit(f"{SCORES_CSV} must contain Sample, PC1, PC2, PC3")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

has_cluster = 'Cluster' in df.columns
if has_cluster:
    uniq = sorted(df['Cluster'].dropna().unique())
    colors = plt.cm.tab10(range(len(uniq)))
    for i, u in enumerate(uniq):
        sub = df[df['Cluster'] == u]
        ax.scatter(sub['PC1'], sub['PC2'], sub['PC3'], s=80, label=f'Cluster {int(u)}', color=colors[i], edgecolor='k')
else:
    ax.scatter(df['PC1'], df['PC2'], df['PC3'], s=80, color='#1f77b4', edgecolor='k')

for _, row in df.iterrows():
    ax.text(row['PC1'], row['PC2'], row['PC3'], str(row['Sample']), fontsize=12)

ax.set_title('PCA Triplot (Major elements): PC1 vs PC2 vs PC3', fontsize=20)
ax.set_xlabel('PC1', fontsize=16)
ax.set_ylabel('PC2', fontsize=16)
ax.set_zlabel('PC3', fontsize=16)
if has_cluster:
    ax.legend(loc='best')
ax.view_init(elev=20, azim=35)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"✓ Saved: {OUT_PNG}")
