import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
from yellowbrick.cluster import KElbowVisualizer


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
            "n_estimators": [5, 10, 25, 50, 100, 200, 300],
            "learning_rate": [0.1, 0.01, 0.001],
            "max_depth": [3, 5, 7, 12],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "gamma": [0.001, 0.01, 0.1, 0, 1, 5],
        }

        random_search = RandomizedSearchCV(
            XGBRegressor(),
            param_grid,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            cv=5,
        )
        random_search.fit(self.X, self.y)
        self.model = random_search.best_estimator_
        self.model.fit(self.X, self.y)

    def get_model(self, model, score=True):
        self.model = model.fit(self.X, self.y)
        if score:
            scores = cross_val_score(
                self.model, self.X, self.y, scoring="neg_mean_squared_error", cv=5
            )
            print(f"RMSE = {np.round(np.sqrt(np.abs(scores.mean())),4)}")
            print(f"STD = {np.round(scores.std(),4)}")

    def load_model(self, path: str):
        self.model = pickle.load(open(path, "rb"))

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("No trained model yet.")
        else:
            pickle.dump(self.model, open(path, "wb"))

    def plot_R2(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(self.y, self.model.predict(self.X), color="blue", alpha=0.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Cross Validation Predictions")
        # Calculate also MAE and R2
        r2 = r2_score(self.y, self.model.predict(self.X))
        ax.text(
            0.95,
            0.05,
            f"R2 = {r2:.2f}",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )
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
            shap.plots.scatter(
                self.shap_values[:, feature], color=self.shap_values[:, color]
            )
        else:
            shap.plots.scatter(self.shap_values[:, feature])

    def supervised_clustering(self, plot=False):
        model = KMeans(n_init="auto")
        visualizer = KElbowVisualizer(model, k=(2, 12))
        visualizer.fit(self.X)
        visualizer.show()
        n_cluster = visualizer.elbow_value_
        print(f"Optimal number of clusters: {n_cluster}")
        kmeans = KMeans(n_clusters=n_cluster, n_init="auto", random_state=42).fit(
            self.shap_df
        )
        self.shap_df["cluster"] = kmeans.labels_
        self.shap_df[self.response_var[0]] = self.y
        self.df["cluster"] = kmeans.labels_

    def study_clusters(self, method="mean", verbose=True):
        if verbose:
            print("Number of elements in each cluster: ")
            print(self.df["cluster"].value_counts())
        if method == "mean":
            if verbose:
                display(self.df.groupby("cluster").mean())
            return self.df.groupby("cluster").mean()
        elif method == "median":
            if verbose:
                display(self.df.groupby("cluster").median())
            return self.df.groupby("cluster").median()
        elif method == "most_frequent":
            if verbose:
                display(self.df.groupby("cluster").agg(lambda x: x.value_counts().index[0]))
            return self.df.groupby("cluster").agg(lambda x: x.value_counts().index[0])
        elif method == "std":
            if verbose:
                display(self.df.groupby("cluster").std())
            return self.df.groupby("cluster").std()
