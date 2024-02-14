from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Linear Regression
linear_param_grid = {
    'fit_intercept': [True, False],
}

# Ridge Regression
ridge_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
    'fit_intercept': [True, False],
}

# Lasso Regression
lasso_param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
    'fit_intercept': [True, False],
}

# K-Nearest Neighbors Regression
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15, 20],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Random Forest Regression
rf_param_grid = {
    'n_estimators': [5, 10, 20, 25, 50],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Support Vector Regression
svr_param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.1, 0.01, 0.001],
    'gamma': [1, 0.1, 0.01, 0.001]
}

# CatBoost Regression
catboost_param_grid = {
    'iterations': [10, 30, 50],
    'learning_rate': [0.01, 0.05, 0.1, 1.0],
    'depth': [4, 6, 8, 10, 12]
}

# LightGBM Regression
lgbm_param_grid = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'num_leaves': [3, 5, 12, 20],
    'learning_rate': [0.01, 0.1, 1.0, 10.0],
    'n_estimators': [5, 10, 20, 50]
}

# XGBoost Regression
xgb_param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 1.0],
    'n_estimators': [10, 25, 30, 50, 100],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5],
    'reg_lambda': [0.0, 0.1, 0.5]
}


# Nested Cross-Validation
def nestedCV(model, p_grid, X, y):
    print(model.__class__.__name__)
    nested_scores = []
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = RandomizedSearchCV(estimator=model, scoring='neg_mean_absolute_error', param_distributions=p_grid, cv=inner_cv, n_iter=100)
    nested_score = cross_val_score(grid, X=X, y=y, scoring='neg_mean_absolute_error', cv=outer_cv)
    nested_scores.append(list(nested_score))
    return model, nested_scores

# Model Selection using NCV
def modelSelection(X_train, y_train):
    nested_catboost, nested_catboost_scores = nestedCV(CatBoostRegressor(verbose=False), catboost_param_grid, X_train, y_train)
    nested_lgbm, nested_lgbm_scores = nestedCV(LGBMRegressor(), lgbm_param_grid, X_train, y_train)
    nested_xgb, nested_xgb_scores = nestedCV(XGBRegressor(n_jobs=-1), xgb_param_grid, X_train, y_train)
    nested_LR, nested_LR_scores = nestedCV(LinearRegression(), linear_param_grid, X_train, y_train)
    nested_ridge, nested_ridge_scores = nestedCV(Ridge(), ridge_param_grid, X_train, y_train)
    nested_lasso, nested_lasso_scores = nestedCV(Lasso(), lasso_param_grid, X_train, y_train)
    nested_knn, nested_knn_scores = nestedCV(KNeighborsRegressor(), knn_param_grid, X_train, y_train)
    nested_rf, nested_rf_scores = nestedCV(RandomForestRegressor(), rf_param_grid, X_train, y_train)
    nested_svr, nested_svr_scores = nestedCV(SVR(), svr_param_grid, X_train, y_train)

    nested_scores_LR = [-item for sublist in nested_LR_scores for item in sublist]
    nested_scores_ridge = [-item for sublist in nested_ridge_scores for item in sublist]
    nested_scores_lasso = [-item for sublist in nested_lasso_scores for item in sublist]
    nested_scores_knn = [-item for sublist in nested_knn_scores for item in sublist]
    nested_scores_rf = [-item for sublist in nested_rf_scores for item in sublist]
    nested_scores_svr = [-item for sublist in nested_svr_scores for item in sublist]
    nested_scores_catboost = [-item for sublist in nested_catboost_scores for item in sublist]
    nested_scores_lgbm = [-item for sublist in nested_lgbm_scores for item in sublist]
    nested_scores_xgb = [-item for sublist in nested_xgb_scores for item in sublist]

    # Combine all nested scores
    nested_scores = [nested_scores_LR, nested_scores_ridge, nested_scores_lasso, nested_scores_knn, nested_scores_rf, nested_scores_svr, nested_scores_catboost, nested_scores_lgbm, nested_scores_xgb]

    # Create a boxplot
    plt.figure(figsize=(10,8))
    plt.boxplot(nested_scores, labels=["LR", "Ridge", "Lasso", "KNN", "RF", "SVR", "CatBoost", "LightGBM", "XGB"])
    plt.xlabel("Models")
    plt.ylabel("Nested Scores")
    plt.title("Nested Scores of All Models")
    plt.show()
    