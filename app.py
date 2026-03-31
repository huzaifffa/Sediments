



import os
import sys
import glob
import pandas as pd
import numpy as np

def ensure_packages():
	try:
		import matplotlib.pyplot as plt
		import seaborn as sns
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import StandardScaler
		from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
	except Exception as e:
		print('Missing package:', e)
		print('Install requirements: pip install pandas numpy matplotlib seaborn scikit-learn scipy')
		sys.exit(1)


def list_csv_files():
	files = sorted(glob.glob('*.csv'))
	return files


def load_data(path):
	df = pd.read_csv(path, index_col=None)
	return df


def sanitize_dataframe(df):
	df = df.copy()
	df.columns = [str(col).replace('\xa0', ' ').strip() for col in df.columns]
	return df


def extract_sample_labels(df):
	label_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
	if label_columns:
		labels = df[label_columns[0]].fillna('').astype(str).str.strip()
		fallback = pd.Series([f'Sample {i + 1}' for i in range(len(df))], index=df.index)
		labels = labels.where(labels != '', fallback)
	else:
		labels = pd.Series([f'Sample {i + 1}' for i in range(len(df))], index=df.index)
	return labels.tolist()


def prepare_analysis_data(df):
	from sklearn.preprocessing import StandardScaler

	df = sanitize_dataframe(df)
	sample_labels = extract_sample_labels(df)
	numeric_df = df.select_dtypes(include=[np.number]).copy()
	numeric_df = numeric_df.dropna(axis=1, how='all')
	if numeric_df.shape[1] < 2:
		raise ValueError('At least two numeric element columns are required for PCA/HCA.')
	if numeric_df.shape[0] < 2:
		raise ValueError('At least two samples are required for PCA/HCA.')

	numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
	numeric_df = numeric_df.fillna(numeric_df.mean())
	numeric_df.index = sample_labels
	scaler = StandardScaler()
	scaled_values = scaler.fit_transform(numeric_df.values)
	scaled_df = pd.DataFrame(scaled_values, index=sample_labels, columns=numeric_df.columns)
	return numeric_df, scaled_df, sample_labels


def compute_pca(scaled_df):
	from sklearn.decomposition import PCA

	n_components = min(scaled_df.shape[0], scaled_df.shape[1])
	pca = PCA(n_components=n_components)
	scores = pca.fit_transform(scaled_df.values)
	columns = [f'PC{i+1}' for i in range(pca.n_components_)]
	loadings = pd.DataFrame(pca.components_.T, index=scaled_df.columns,
							columns=[f'PC{i+1}' for i in range(pca.n_components_)])
	explained = pd.Series(pca.explained_variance_ratio_, index=loadings.columns)
	scores_df = pd.DataFrame(scores, index=scaled_df.index, columns=columns)
	return pca, scores_df, loadings, explained


def compute_hca_clusters(scaled_df, n_clusters=4, method='ward'):
	from scipy.cluster.hierarchy import linkage, fcluster

	linkage_matrix = linkage(scaled_df.values, method=method)
	clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
	cluster_series = pd.Series(clusters, index=scaled_df.index, name='Cluster')
	return linkage_matrix, cluster_series


def plot_heatmap(scaled_df, cluster_series, linkage_matrix, outpath):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from matplotlib.colors import LinearSegmentedColormap
	from scipy.cluster.hierarchy import linkage, fcluster

	heatmap_min = float(scaled_df.min().min())
	heatmap_max = float(scaled_df.max().max())
	if np.isclose(heatmap_max, heatmap_min):
		heatmap_df = pd.DataFrame(0.5, index=scaled_df.index, columns=scaled_df.columns)
	else:
		heatmap_df = (scaled_df - heatmap_min) / (heatmap_max - heatmap_min)

	cluster_palette = ['#1f4e79', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#b07aa1']
	heatmap_cmap = LinearSegmentedColormap.from_list(
		'blue_green_yellow_heatmap',
		['#08306b', '#2171b5', '#41b6c4', '#4daf4a', '#a1d76a', '#ffe066']
	)
	title_fontsize = 26
	axis_fontsize = 24
	tick_fontsize = 16
	colorbar_fontsize = 16
	cluster_band_size = 0.025
	unique_clusters = sorted(cluster_series.unique())
	cluster_colors = {
		cluster_id: cluster_palette[(cluster_id - 1) % len(cluster_palette)]
		for cluster_id in unique_clusters
	}
	row_colors = cluster_series.map(cluster_colors)
	column_linkage_matrix = linkage(scaled_df.T.values, method='ward')
	column_cluster_ids = fcluster(
		column_linkage_matrix,
		t=min(4, scaled_df.shape[1]),
		criterion='maxclust'
	)
	column_cluster_colors = {
		cluster_id: cluster_palette[(cluster_id - 1) % len(cluster_palette)]
		for cluster_id in sorted(np.unique(column_cluster_ids))
	}
	col_colors = pd.Series(column_cluster_ids, index=scaled_df.columns).map(column_cluster_colors)

	cluster_grid = sns.clustermap(
		heatmap_df,
		row_linkage=linkage_matrix,
		col_linkage=column_linkage_matrix,
		row_colors=row_colors,
		col_colors=col_colors,
		cmap=heatmap_cmap,
		vmin=0,
		vmax=1,
		linewidths=0.2,
		linecolor='#f4f0e8',
		figsize=(16, max(9, len(scaled_df) * 0.65)),
		dendrogram_ratio=(0.16, 0.14),
		colors_ratio=(cluster_band_size, cluster_band_size),
		cbar_pos=(0.93, 0.24, 0.025, 0.48),
		cbar_kws={'label': 'Normalized Concentration (0-1)'}
	)
	cluster_grid.fig.suptitle('Heatmap of Trace Elements', fontsize=title_fontsize, y=0.99, fontweight='bold')
	cluster_grid.ax_heatmap.set_xlabel('Elements', fontsize=axis_fontsize)
	cluster_grid.ax_heatmap.set_ylabel('Samples', fontsize=axis_fontsize)
	cluster_grid.ax_heatmap.tick_params(axis='x', labelsize=tick_fontsize, pad=8)
	cluster_grid.ax_heatmap.tick_params(axis='y', labelsize=tick_fontsize, pad=6)
	plt.setp(cluster_grid.ax_heatmap.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')
	plt.setp(cluster_grid.ax_heatmap.get_yticklabels(), rotation=0)
	cluster_grid.ax_row_dendrogram.set_xticks([])
	cluster_grid.ax_row_dendrogram.set_yticks([])
	cluster_grid.ax_col_dendrogram.set_xticks([])
	cluster_grid.ax_col_dendrogram.set_yticks([])
	cluster_grid.ax_row_colors.set_xticks([])
	cluster_grid.ax_row_colors.set_yticks([])
	cluster_grid.ax_col_colors.set_xticks([])
	cluster_grid.ax_col_colors.set_yticks([])
	cluster_grid.cax.tick_params(labelsize=colorbar_fontsize)
	cluster_grid.cax.set_ylabel('Normalized Concentration (0-1)', fontsize=colorbar_fontsize)
	cluster_grid.fig.subplots_adjust(left=0.08, right=0.9, bottom=0.18, top=0.9)
	cluster_grid.fig.savefig(outpath, dpi=300, bbox_inches='tight')
	plt.close(cluster_grid.fig)


def plot_scree(explained, outpath):
	import matplotlib.pyplot as plt

	pcs = np.arange(1, len(explained) + 1)
	variance_pct = explained.values * 100
	cumulative_pct = np.cumsum(variance_pct)

	fig, ax1 = plt.subplots(figsize=(9, 5))
	ax1.bar(pcs, variance_pct, color='#4c78a8', alpha=0.85)
	ax1.plot(pcs, variance_pct, color='#1f1f1f', marker='o', linewidth=1.5)
	ax1.set_xlabel('Principal Component')
	ax1.set_ylabel('Explained Variance (%)')
	ax1.set_xticks(pcs)
	ax1.set_title('PCA Scree Plot')

	ax2 = ax1.twinx()
	ax2.plot(pcs, cumulative_pct, color='#f58518', marker='s', linewidth=2)
	ax2.set_ylabel('Cumulative Variance (%)')
	ax2.set_ylim(0, 105)

	for pc, value in zip(pcs, variance_pct):
		ax1.text(pc, value + 0.6, f'{value:.1f}%', ha='center', va='bottom', fontsize=8)

	fig.tight_layout()
	plt.savefig(outpath, dpi=300, bbox_inches='tight')
	plt.close(fig)


def plot_triplot(scores, cluster_series, explained, outpath):
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

	title_fontsize = 22
	axis_fontsize = 14
	tick_fontsize = 13
	label_fontsize = 13
	legend_fontsize = 13

	if scores.shape[1] < 3:
		raise ValueError('At least three principal components are required for the triplot.')

	x_col = 'PC2'
	y_col = 'PC3'
	z_col = 'PC1'
	colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2']

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	legend_handles = []

	for cluster_id in sorted(cluster_series.unique()):
		mask = cluster_series == cluster_id
		cluster_scores = scores.loc[mask]
		color = colors[(cluster_id - 1) % len(colors)]
		ax.scatter(
			cluster_scores[x_col],
			cluster_scores[y_col],
			cluster_scores[z_col],
			s=45,
			color=color,
			edgecolors='black',
			linewidths=0.4,
			alpha=0.9
		)
		legend_handles.append(
			Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster_id}',
				markerfacecolor=color, markeredgecolor='black', markersize=10)
		)

	for sample_name, row in scores.iterrows():
		ax.text(row[x_col], row[y_col], row[z_col], sample_name, fontsize=label_fontsize)

	ax.set_xlabel(f'{x_col} ({explained[x_col] * 100:.1f}%)', fontsize=axis_fontsize, labelpad=12)
	ax.set_ylabel(f'{y_col} ({explained[y_col] * 100:.1f}%)', fontsize=axis_fontsize, labelpad=12)
	ax.set_zlabel(f'{z_col} ({explained[z_col] * 100:.1f}%)', fontsize=axis_fontsize, labelpad=12)
	ax.set_title('PCA Triplot (PC2-PC3-PC1) of Major Elements', fontsize=title_fontsize, pad=18)
	ax.tick_params(axis='x', labelsize=tick_fontsize)
	ax.tick_params(axis='y', labelsize=tick_fontsize)
	ax.tick_params(axis='z', labelsize=tick_fontsize)
	ax.view_init(elev=22, azim=132)
	ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=legend_fontsize)
	plt.tight_layout()
	plt.savefig(outpath, dpi=300, bbox_inches='tight')
	plt.close()


def plot_dendrogram(linkage_matrix, labels, outpath):
	import matplotlib.pyplot as plt
	from scipy.cluster.hierarchy import dendrogram

	title_fontsize = 20
	axis_fontsize = 16
	tick_fontsize = 13
	leaf_fontsize = 13

	plt.figure(figsize=(12, 6))
	dendrogram(linkage_matrix, labels=labels, leaf_rotation=45, leaf_font_size=leaf_fontsize, color_threshold=None)
	plt.title('Hierarchical Clustering Dendrogram of Trace Elements', fontsize=title_fontsize)
	plt.xlabel('Samples', fontsize=axis_fontsize)
	plt.ylabel('Distance', fontsize=axis_fontsize)
	plt.xticks(fontsize=tick_fontsize)
	plt.yticks(fontsize=tick_fontsize)
	plt.tight_layout()
	plt.savefig(outpath, dpi=300, bbox_inches='tight')
	plt.close()


# def save_numeric_outputs(outdir, prefix, scores, loadings, explained, cluster_series):
# 	scores.to_csv(os.path.join(outdir, f'{prefix}_pca_scores.csv'))
# 	loadings.to_csv(os.path.join(outdir, f'{prefix}_pca_loadings.csv'))
# 	explained.to_csv(
# 		os.path.join(outdir, f'{prefix}_pca_explained_variance.csv'),
# 		header=['explained_variance_ratio']
# 	)
# 	cluster_series.to_csv(os.path.join(outdir, f'{prefix}_hca_clusters.csv'), header=True)


def output_prefix(csv_path):
	base = os.path.splitext(os.path.basename(csv_path))[0].lower()
	return ''.join(ch if ch.isalnum() else '_' for ch in base).strip('_') or 'analysis'


def ensure_outputs_dir(path='outputs'):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)
	return path


def run_pca_menu():
	ensure_packages()
	csvs = list_csv_files()
	if not csvs:
		print('No CSV files found in the current directory.')
		return

	print('CSV files found:')
	for i, f in enumerate(csvs, 1):
		print(f'{i}. {f}')
	sel = input('Select file number to run PCA on (default 1): ').strip()
	try:
		idx = int(sel) - 1 if sel else 0
		csv_path = csvs[idx]
	except Exception:
		print('Invalid choice, using first file.')
		csv_path = csvs[0]

	df = load_data(csv_path)
	try:
		numeric_df, scaled_df, sample_labels = prepare_analysis_data(df)
		pca, scores, loadings, explained = compute_pca(scaled_df)
		linkage_matrix, cluster_series = compute_hca_clusters(scaled_df, n_clusters=4)
	except ValueError as exc:
		print(f'Analysis error: {exc}')
		return

	outdir = ensure_outputs_dir('outputs')
	prefix = output_prefix(csv_path)
	# save_numeric_outputs(outdir, prefix, scores, loadings, explained, cluster_series)
	print(f'Processed {len(sample_labels)} samples and {numeric_df.shape[1]} element columns.')

	print('\nWhat would you like to calculate/plot?')
	print('1. heatmap')
	print('2. triplot')
	print('3. dendrogram')
	print('4. scree plot')
	print('5. all')
	print('6. exit')
	choice = input('Enter choice (1-6): ').strip()

	if choice == '1' or choice == '5':
		outpath = os.path.join(outdir, f'{prefix}_hca_cluster_heatmap.png')
		plot_heatmap(scaled_df, cluster_series, linkage_matrix, outpath)
		print('Saved heatmap to', outpath)

	if choice == '2' or choice == '5':
		outpath = os.path.join(outdir, f'{prefix}_pca_triplot.png')
		plot_triplot(scores, cluster_series, explained, outpath)
		print('Saved triplot to', outpath)

	if choice == '3' or choice == '5':
		outpath = os.path.join(outdir, f'{prefix}_hca_dendrogram.png')
		plot_dendrogram(linkage_matrix, sample_labels, outpath)
		print('Saved dendrogram to', outpath)

	if choice == '4' or choice == '5':
		outpath = os.path.join(outdir, f'{prefix}_pca_scree_plot.png')
		plot_scree(explained, outpath)
		print('Saved scree plot to', outpath)

	print('PCA outputs saved in', outdir)


if __name__ == '__main__':
	run_pca_menu()
