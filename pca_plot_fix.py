import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')

CSV_PATH = 'pca_scores.csv'
OUT_PATH = 'pca_biplot_pc1_pc3_fixed.png'

if not os.path.exists(CSV_PATH):
    raise SystemExit(f"{CSV_PATH} not found in workspace. Run PCA first or place the file here.")

df = pd.read_csv(CSV_PATH)

if not {'PC1', 'PC3', 'Sample'}.issubset(df.columns):
    raise SystemExit('pca_scores.csv must contain at least the columns: Sample, PC1, PC3')

has_cluster = 'Cluster' in df.columns

fig, ax = plt.subplots(figsize=(12, 9))

if has_cluster:
    sns.scatterplot(data=df, x='PC1', y='PC3', hue='Cluster', palette='tab10', s=80, edgecolor='k', ax=ax)
else:
    sns.scatterplot(data=df, x='PC1', y='PC3', color='#1f77b4', s=80, edgecolor='k', ax=ax)

# Annotate sample labels with small offset to avoid overlap with markers
for _, row in df.iterrows():
    ax.annotate(row['Sample'], (row['PC1'], row['PC3']), xytext=(4, 3), textcoords='offset points', fontsize=8)

ax.set_title('PCA Biplot: PC1 vs PC3', fontsize=16, fontweight='bold')
ax.set_xlabel('PC1', fontsize=13)
ax.set_ylabel('PC3', fontsize=13)

# Improve layout so labels are not cut off
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.12)

# Save with bbox_inches='tight' and a small pad
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"✓ Saved: {OUT_PATH}")

plt.show()
