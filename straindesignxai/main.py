import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

class DataLoader:
    def __init__(self, input_csv: str, target: str):
        self.input_csv = input_csv
        self.target = target
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.shap_values = None
        self.shap_df = None
        self.__get_data()
                
    def __get_data(self):
        _ = pd.read_csv(self.input_csv)
        cols = _['Type'].unique()
        idx = _['Line Name'].unique()
        self.df = pd.DataFrame(index=idx, columns=cols)
        for i in idx:
            for j in cols:
                self.df.loc[i, j] = _[(_['Line Name'] == i) & (_['Type'] == j)]['Value'].values[0]
        self.df = self.df.astype(float)
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]
    
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
        self.df['cluster'] = self.shap_df['cluster']
        
    def __str__(self):
        return f"Dataset size: {self.df.shape}"
    
    

