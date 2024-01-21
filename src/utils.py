import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm

def plot_corr_heatmap(df: pd.DataFrame, title: str) -> None:
    ''' Calculate corr matrix and plot heatmap '''
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    plt.title(title)
    plt.show()

def plot_pca(pca_df: pd.DataFrame, pca: PCA) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(pca_df['PC1'], pca_df['PC2'], s=pca_df['Value']*100, color='blue', marker='x', alpha=0.5)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].tick_params(labelsize=8)
    axes[0].set_title('Scatter Plot')
    points_to_label = pca_df[pca_df['Value'] > pca_df['Value'].quantile(0.9)]
    for i in range(len(points_to_label)):
        axes[0].text(points_to_label.iloc[i, 0], points_to_label.iloc[i, 1], points_to_label.index[i], fontsize=8)    
    pc_var = pca.explained_variance_ratio_
    pc_var = np.round(pc_var * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(pc_var)+1)]
    axes[1].bar(x=range(1, len(pc_var)+1), height=pc_var, tick_label=labels)
    axes[1].set_ylabel('Percentage of Variance Explained')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_title('Percentage of Variance Explained')
    plt.tight_layout()
    plt.show()

def tsne_analysis(df: pd.DataFrame, target_col='Value', n_components=2, perplexity=12) -> pd.DataFrame:
    ''' Perform TSNE analysis and plot results'''
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_df = pd.DataFrame(tsne.fit_transform(df.drop(target_col, axis=1)))
    tsne_df.index = df.index
    # Plot TSNE1 vs TSNE2 with labels; control marker size
    plt.scatter(tsne_df[0], tsne_df[1], s=8, color='black')
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.rcParams.update({'font.size': 8})
    for i, txt in enumerate(tsne_df.index):
        plt.annotate(txt, (tsne_df[0][i], tsne_df[1][i]))
    plt.show()
    tsne_df.columns = ['TSNE1', 'TSNE2']
    return tsne_df

def nested_cv(model, p_grid, X, y, n_trials = 3):
    nested_scores = []
    for i in tqdm(range(n_trials)):
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=model, scoring='neg_mean_absolute_error', param_grid=p_grid, 
                                 cv=inner_cv)
        nested_score = cross_val_score(clf, X=X, y=y, 
                                       scoring='neg_mean_absolute_error', cv=outer_cv)
        nested_scores.append(list(nested_score))
    return clf, nested_scores

def plot_pred_vs_actual(y_true, y_pred, model_name) -> None:
    # Plot true vs predicted values
    plt.scatter(y_true, y_pred)
    plt.xlim(-50, 200)
    plt.ylim(-100, 250)
    plt.plot([-50, 200], [-50, 200], 'k--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.text(-40, 220, f'R2: {r2:.2f}')
    plt.text(-40, 200, f'MAE: {mae:.2f}')
    plt.title(model_name)
    plt.show()

def cv_on_whole_train_set(data, model, input, response) -> None:
    X = data[input]
    y = data[response]
    y_pred = cross_val_predict(model, X, y, cv=5)
    plot_pred_vs_actual(y, y_pred, 'CV')

