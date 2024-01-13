import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
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
        
    def train_xgb(self, n_iter=100):
        param_grid = {
            'n_estimators': [5, 10, 25, 50, 100, 200, 300],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 7, 12],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0.001, 0.01, 0.1, 0, 1, 5]
        }

        random_search = RandomizedSearchCV(XGBRegressor(), param_grid, n_iter=n_iter, 
                                           scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
        random_search.fit(self.X, self.y)
        self.model = random_search.best_estimator_
        self.model.fit(self.X, self.y)

                
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
        
    def pdplot(self, feature: str, ice=False):
        shap.partial_dependence_plot(
            feature,
            self.model.predict,
            self.X,
            ice=ice,
            model_expected_value=True,
            feature_expected_value=True,
        )
    
    def get_shap_values(self, plot=True):
        explainer = shap.TreeExplainer(self.model, self.X)
        self.shap_values = explainer(self.X)
        self.shap_df = pd.DataFrame(self.shap_values.values, columns=self.X.columns)
        if plot:
            self.__plot_shap_summary()

    def __plot_shap_summary(self):
        shap.summary_plot(self.shap_values, self.X, plot_type="bar")
        shap.summary_plot(self.shap_values, self.X)
        shap.plots.heatmap(self.shap_values, instance_order=self.shap_values.sum(1))
        plt.show()
    
    def shap_scatter(self, feature: str, color=None):
        if color:
            shap.plots.scatter(self.shap_values[:, feature], color=self.shap_values[:, color])
        else:
            shap.plots.scatter(self.shap_values[:, feature])
                
    def supervised_clustering(self, plot=False):
        model = KMeans(n_init='auto')
        visualizer = KElbowVisualizer(model, k=(2,12))
        visualizer.fit(self.X)
        visualizer.show()
        n_cluster = visualizer.elbow_value_
        print(f"Optimal number of clusters: {n_cluster}")
        kmeans = KMeans(n_clusters=n_cluster, n_init='auto', random_state=42).fit(self.shap_df)
        self.shap_df['cluster'] = kmeans.labels_
        self.shap_df[self.response_var[0]] = self.y
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
            
    
    

