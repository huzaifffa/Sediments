import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

sns.set(style='whitegrid')

CSV_PATH = 'pca_scores.csv'
OUT_PATH = 'pca_triplot_3d_fixed.png'

if not os.path.exists(CSV_PATH):
    raise SystemExit(f"{CSV_PATH} not found. Place the file in the workspace and rerun.")

df = pd.read_csv(CSV_PATH)
required = {'Sample', 'PC1', 'PC2', 'PC3'}
if not required.issubset(df.columns):
    raise SystemExit(f"{CSV_PATH} must contain columns: {', '.join(required)}")

has_cluster = 'Cluster' in df.columns

fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111, projection='3d')

if has_cluster:
    clusters = df['Cluster'].astype(str)
    uniq = clusters.unique()
    palette = sns.color_palette('tab10', n_colors=len(uniq))
    for i, u in enumerate(uniq):
        sub = df[clusters == u]
        ax.scatter(sub['PC1'], sub['PC2'], sub['PC3'], s=80, label=f'Cluster {u}',
                   color=palette[i], edgecolor='k')
else:
    ax.scatter(df['PC1'], df['PC2'], df['PC3'], s=80, color='#1f77b4', edgecolor='k')

# Add sample labels near points
for _, row in df.iterrows():
    ax.text(row['PC1'], row['PC2'], row['PC3'], str(row['Sample']), fontsize=8)

ax.set_title('PCA Triplot: PC1 vs PC2 vs PC3', fontsize=18, fontweight='bold')
ax.set_xlabel('PC1', fontsize=14)
ax.set_ylabel('PC2', fontsize=14)
ax.set_zlabel('PC3', fontsize=14)

if has_cluster:
    ax.legend(loc='best', fontsize=11)

ax.grid(True)
ax.view_init(elev=20, azim=35)

# Improve layout and save large figure without clipping
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"✓ Saved: {OUT_PATH}")
plt.show()
