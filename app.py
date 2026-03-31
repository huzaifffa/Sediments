



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
		from scipy.cluster.hierarchy import dendrogram, linkage
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


def compute_pca(df, n_components=3):
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler

	X = df.select_dtypes(include=[np.number]).copy()
	X = X.fillna(X.mean())
	scaler = StandardScaler()
	Xs = scaler.fit_transform(X.values)
	pca = PCA(n_components=n_components)
	scores = pca.fit_transform(Xs)
	loadings = pd.DataFrame(pca.components_.T, index=X.columns,
							columns=[f'PC{i+1}' for i in range(pca.n_components_)])
	explained = pd.Series(pca.explained_variance_ratio_, index=loadings.columns)
	scores_df = pd.DataFrame(scores, columns=loadings.columns)
	return pca, scores_df, loadings, explained


def plot_heatmap(loadings, outpath):
	import matplotlib.pyplot as plt
	import seaborn as sns

	plt.figure(figsize=(8, max(4, len(loadings) * 0.2)))
	sns.heatmap(loadings, annot=True, cmap='coolwarm')
	plt.title('PCA Loadings Heatmap')
	plt.tight_layout()
	plt.savefig(outpath)
	plt.close()


def plot_triplot(scores, outpath):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(scores.iloc[:, 1], scores.iloc[:, 0], scores.iloc[:, 2], s=30)
	ax.set_xlabel(scores.columns[1])
	ax.set_ylabel(scores.columns[0])
	ax.set_zlabel(scores.columns[2])
	ax.set_title('PCA Triplot (PC2-PC1-PC3)')
	plt.tight_layout()
	plt.savefig(outpath)
	plt.close()


def plot_dendrogram(scores, outpath):
	import matplotlib.pyplot as plt
	from scipy.cluster.hierarchy import linkage, dendrogram

	Z = linkage(scores.values, method='ward')
	plt.figure(figsize=(10, 6))
	dendrogram(Z, no_labels=True, color_threshold=None)
	plt.title('Hierarchical Clustering Dendrogram (on PCA scores)')
	plt.tight_layout()
	plt.savefig(outpath)
	plt.close()


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
	pca, scores, loadings, explained = compute_pca(df, n_components=3)

	outdir = ensure_outputs_dir('outputs')
	# save numeric results
	# scores.to_csv(os.path.join(outdir, 'pca_scores.csv'), index=False)
	# loadings.to_csv(os.path.join(outdir, 'pca_loadings.csv'))
	# explained.to_csv(os.path.join(outdir, 'pca_explained_variance.csv'), header=['explained_variance_ratio'])

	print('\nWhat would you like to calculate/plot in PCA?')
	print('1. heatmap')
	print('2. triplot')
	print('3. dendrogram')
	print('4. all')
	print('5. exit')
	choice = input('Enter choice (1-5): ').strip()

	if choice == '1' or choice == '4':
		outpath = os.path.join(outdir, 'pca_heatmap.png')
		plot_heatmap(loadings, outpath)
		print('Saved heatmap to', outpath)

	if choice == '2' or choice == '4':
		outpath = os.path.join(outdir, 'pca_triplot.png')
		plot_triplot(scores, outpath)
		print('Saved triplot to', outpath)

	if choice == '3' or choice == '4':
		outpath = os.path.join(outdir, 'pca_dendrogram.png')
		plot_dendrogram(scores, outpath)
		print('Saved dendrogram to', outpath)

	print('PCA outputs saved in', outdir)


if __name__ == '__main__':
	run_pca_menu()
