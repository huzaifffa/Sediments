import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# If an Excel file with multiple sheets exists, process each sheet separately.
INPUT_XLSX = 'WetlandSoil.xlsx'
INPUT_CSV = 'WetlandSoil.csv'

def safe_name(name: str) -> str:
    return str(name).replace(' ', '_').replace('/', '_')

def analyze_df(df: pd.DataFrame, sheet_name: str):
    print(f"\n--- Processing sheet: {sheet_name} ---")
    # Identify sample/name column
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric_cols) == 0:
        raise ValueError('No non-numeric column found for sample names')
    sample_col = non_numeric_cols[0]

    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(features) == 0:
        raise ValueError('No numeric features found for PCA')

    mask = ~df[features].isnull().any(axis=1)
    df_num = df.loc[mask, features].astype(float)
    samples = df.loc[mask, sample_col].astype(str).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num.values)

    # PCA
    n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    base = safe_name(sheet_name)
    # Save explained variance
    explained = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'ExplainedVarianceRatio': pca.explained_variance_ratio_,
        'Cumulative': np.cumsum(pca.explained_variance_ratio_)
    })
    explained.to_csv(f'pca_explained_variance_{base}.csv', index=False)
    print(f'\u2713 Saved: pca_explained_variance_{base}.csv')

    # Loadings and scores
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    loadings.to_csv(f'pca_loadings_{base}.csv')
    scores_df = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    scores_df.insert(0, 'Sample', samples)
    scores_df.to_csv(f'pca_scores_{base}.csv', index=False)
    print(f'\u2713 Saved: pca_loadings_{base}.csv, pca_scores_{base}.csv')

    # Scree
    plt.figure()
    plt.bar(explained['PC'], explained['ExplainedVarianceRatio'], color='C0')
    plt.plot(explained['PC'], explained['Cumulative'], color='C1', marker='o')
    plt.ylabel('Explained variance')
    plt.title(f'PCA Scree Plot ({sheet_name})')
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'pca_scree_{base}.png', dpi=300)
    plt.close()

    # PC1 vs PC2
    if pca.n_components_ >= 2:
        x = scores[:, 0]
        y = scores[:, 1]
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, c='C0')
        for i, txt in enumerate(samples):
            plt.text(x[i], y[i], txt, fontsize=12)
        plt.xlabel('PC1', fontsize=16)
        plt.ylabel('PC2', fontsize=16)
        plt.title(f'PCA: PC1 vs PC2 ({sheet_name})', fontsize=18)
        plt.tight_layout()
        plt.savefig(f'pca_PC1_vs_PC2_{base}.png', dpi=300)
        plt.close()

        # Biplot-like
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, c='C0', alpha=0.6)
        x_scale = (max(x) - min(x)) if max(x) != min(x) else 1.0
        y_scale = (max(y) - min(y)) if max(y) != min(y) else 1.0
        for i, feat in enumerate(features):
            dx = loadings.iloc[i, 0] * x_scale * 0.6
            dy = loadings.iloc[i, 1] * y_scale * 0.6
            plt.arrow(0, 0, dx, dy, head_width=0.02 * max(x_scale, y_scale), head_length=0.02 * max(x_scale, y_scale), linewidth=1, color='red')
            plt.text(dx * 1.15, dy * 1.15, feat, color='red', fontsize=12)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'PCA Biplot (approx) ({sheet_name})')
        plt.tight_layout()
        plt.savefig(f'pca_biplot_{base}.png', dpi=300)
        plt.close()

    # Hierarchical Clustering (Ward) and dendrogram
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    Z = linkage(X_scaled, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=samples, leaf_rotation=90, leaf_font_size=8)
    plt.title(f'Hierarchical Clustering Dendrogram (Ward) - {sheet_name}')
    plt.tight_layout()
    plt.savefig(f'dendrogram_ward_{base}.png', dpi=300)
    plt.close()

    # Cut into clusters (choose 4 as default)
    clusters = fcluster(Z, 4, criterion='maxclust')
    result_df = pd.DataFrame({'Sample': samples, 'Cluster': clusters})
    result_df.to_csv(f'cluster_assignments_{base}.csv', index=False)

    # Heatmap sorted by cluster
    sorted_idx = np.argsort(clusters)
    X_sorted = X_scaled[sorted_idx]
    names_sorted = samples[sorted_idx]

    plt.figure(figsize=(10, max(4, 0.25 * len(names_sorted))))
    im = plt.imshow(X_sorted, aspect='auto', cmap='RdYlBu_r')
    plt.colorbar(im, label='Standardized Value')
    plt.yticks(range(len(names_sorted)), names_sorted, fontsize=10)
    plt.xticks(range(len(features)), features, rotation=90, fontsize=10)
    plt.title(f'Heatmap (sorted by cluster) - {sheet_name}', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'heatmap_clusters_{base}.png', dpi=300)
    plt.close()

    # Distance matrix heatmap
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))
    plt.figure(figsize=(10, 10))
    im = plt.imshow(dist_matrix, cmap='YlOrRd')
    plt.colorbar(im, label='Euclidean distance')
    plt.xticks(range(len(samples)), samples, rotation=90, fontsize=10)
    plt.yticks(range(len(samples)), samples, fontsize=10)
    plt.title(f'Distance matrix - {sheet_name}', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'distance_matrix_{base}.png', dpi=300)
    plt.close()

    print(f'Completed analysis for sheet: {sheet_name}')


# Main flow: check for Excel first, else CSV
if os.path.exists(INPUT_XLSX):
    print(f'Found Excel file: {INPUT_XLSX} - processing all sheets')
    xls = pd.read_excel(INPUT_XLSX, sheet_name=None)
    for sheet_name, df_sheet in xls.items():
        analyze_df(df_sheet, sheet_name)
elif os.path.exists(INPUT_CSV):
    print(f'Excel not found. Processing {INPUT_CSV} as single sheet')
    df = pd.read_csv(INPUT_CSV)
    analyze_df(df, 'Sheet1')
else:
    raise FileNotFoundError('Neither WetlandSoil.xlsx nor WetlandSoil.csv found')

print('\nAll done.')
