import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from IPython.display import display

class DataLoader:
    def __init__(self, input_csv: str, input_vars: list, response_var: list):
        self.input_csv = input_csv
        self.input_var = input_vars
        self.response_var = response_var
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.shap_values = None
        self.shap_df = None
        self.__get_data_edd()
    
    def __get_data_edd(self):
        self.df = pd.read_csv(self.input_csv, index_col=0)
        self.df = self.df[self.input_var + self.response_var]
        self.X = self.df[self.input_var]
        self.y = self.df[self.response_var].values.ravel()
        print(f"Dataset size: {self.df.shape}")
                
    def get_model(self, model, score=True):
        self.model = model.fit(self.X, self.y)
        if score:
            scores = cross_val_score(self.model, self.X, self.y, scoring='neg_mean_squared_error', cv=5)
            print(f'RMSE = {np.round(np.sqrt(np.abs(scores.mean())),4)}')
            print(f'STD = {np.round(scores.std(),4)}')
    
    def plot_R2(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(self.y, self.model.predict(self.X), color='blue', alpha=0.5)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Cross Validation Predictions')
        plt.show()
        
    def get_shap_values(self, plot=True):
        explainer = shap.TreeExplainer(self.model, self.X)
        self.shap_values = explainer(self.X)
        self.shap_df = pd.DataFrame(self.shap_values.values, columns=self.X.columns)
        if plot:
            shap.summary_plot(self.shap_values, self.X, plot_type="bar")
            shap.summary_plot(self.shap_values, self.X)
            order = np.argsort(self.model.predict(self.X))
            shap.plots.heatmap(self.shap_values, instance_order=order)
            plt.show()
            
    def supervised_clustering(self, plot=False):
        model = KMeans(n_init='auto')
        visualizer = KElbowVisualizer(model, k=(2,12))
        visualizer.fit(self.X)
        visualizer.show()
        n_cluster = visualizer.elbow_value_
        print(f"Optimal number of clusters: {n_cluster}")
        kmeans = KMeans(n_clusters=n_cluster, n_init='auto', random_state=42).fit(self.shap_df)
        self.shap_df['cluster'] = kmeans.labels_
        self.shap_df['y'] = self.y
        self.df['cluster'] = kmeans.labels_
        
    def study_clusters(self, method='mean'):
        print('Number of elements in each cluster: ')
        print(self.df['cluster'].value_counts())
        if method == 'mean':
            display(self.df.groupby('cluster').mean())
        elif method == 'median':
            display(self.df.groupby('cluster').median())
        elif method == 'most_frequent':
            display(self.df.groupby('cluster').agg(lambda x: x.value_counts().index[0]))
        elif method == 'std':
            display(self.df.groupby('cluster').std())
            
            
    

