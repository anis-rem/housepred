import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from prePro import processed_data

scaler = StandardScaler()
pd.set_option('display.max_columns', None)

df = processed_data
print(df.head())

df["total rooms"] = df["bathrooms"] + df["bedrooms"]
df["room per floor"] = df["total rooms"] / df["stories"].replace(0, 1)
df['yess score'] = (
    df['mainroad'] + df['guestroom'] + df['basement'] +
    df['hotwaterheating'] + df['airconditioning'] + df['prefarea']
)

print(df.head(1))
print("missing values:")
print(df.isnull().sum())

df["area"] *= 1.3
df["total rooms"] *= 1.2
df["room per floor"] *= 1.1
df["yess score"] *= 1.5

x = df.drop(columns=["price"])
x_scaled = scaler.fit_transform(x)
y = df["price"]

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2)

print("testing data:", xtest.shape)
print("training data:", xtrain.shape)

lr_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
ridge_model = Ridge(tol=1e-3)
lasso_model = Lasso(tol=1e-3)
elastic_model = ElasticNet(tol=1e-3)

models = [ridge_model, lasso_model, elastic_model, lr_model, tree_model, rf_model]
models_no_linear = [ridge_model, lasso_model, elastic_model, tree_model, rf_model]

def get_param_grid(model):
    name = type(model).__name__

    if name == 'RandomForestRegressor':
        return {
            "n_estimators": [200, 300, 500, 800, 1000],
            "max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7],
            "max_depth": [15, 20, 25, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8, 12],
            "bootstrap": [True, False],
            "criterion": ["squared_error", "absolute_error", "friedman_mse"],
            "min_weight_fraction_leaf": [0.0, 0.01, 0.02],
            "max_leaf_nodes": [None, 100, 200, 500],
            "min_impurity_decrease": [0.0, 0.001, 0.005, 0.01],
            "max_samples": [None, 0.7, 0.8, 0.9],
            "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02]
        }

    if name == 'DecisionTreeRegressor':
        return {
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10, 20, 50],
            "min_samples_leaf": [1, 2, 4, 8, 12, 20],
            "criterion": ["squared_error", "absolute_error", "friedman_mse"],
            "splitter": ["best", "random"],
            "max_features": ["sqrt", "log2", None],
            "min_weight_fraction_leaf": [0.0, 0.01, 0.02, 0.05],
            "max_leaf_nodes": [None, 50, 100, 200],
            "min_impurity_decrease": [0.0, 0.001, 0.005, 0.01],
            "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02]
        }

    if name == 'Ridge':
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "fit_intercept": [True, False],
            "solver": ["auto", "svd", "cholesky", "sparse_cg", "sag", "saga"],
            "max_iter": [1000, 2000, 10000, 20000]
        }

    if name == 'Lasso':
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True, False],
            "precompute": [True, False],
            "max_iter": [1000, 2000, 5000, 10000],
            "tol": [1e-4, 1e-3, 1e-2],
            "positive": [True, False],
            "selection": ["cyclic", "random"]
        }

    if name == 'ElasticNet':
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            "fit_intercept": [True, False],
            "precompute": [True, False],
            "max_iter": [1000, 2000, 5000, 10000],
            "tol": [1e-4, 1e-3, 1e-2],
            "positive": [True, False],
            "selection": ["cyclic", "random"]
        }

    if name == 'LinearRegression':
        return {
            "fit_intercept": [True, False],
            "positive": [True, False]
        }

    return {}

def evaluate_models(models, xtrain, xtest, ytrain, ytest):
    best_model = None
    best_r2 = 0

    for model in models:
        print(f"\n{type(model).__name__} results:")
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        mae = mean_absolute_error(ytest, ypred)
        mse = mean_squared_error(ytest, ypred)
        r2 = r2_score(ytest, ypred)

        print("mae:", round(mae, 2))
        print("mse:", round(mse, 2))
        print("r² score:", round(r2, 4))

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    return best_model, best_r2

def optimize_best_model(best_model, xtrain, ytrain, xtest, ytest):
    name = type(best_model).__name__
    print(f"\noptimizing {name}...")
    param_grid = get_param_grid(best_model)

    random_search = RandomizedSearchCV(
        estimator=type(best_model)(),
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        verbose=1,
        n_jobs=-1,
        scoring='r2',
        random_state=42
    )

    random_search.fit(xtrain, ytrain)
    best = random_search.best_estimator_
    ypred = best.predict(xtest)
    r2 = r2_score(ytest, ypred)

    print(f"\noptimized {name} results:")
    print("best params:", random_search.best_params_)
    print("test r² score:", round(r2, 4))

    return best, random_search.best_params_

def vs_linear(lr_model, models, xtrain, xtest, ytrain, ytest):
    lr_model.fit(xtrain, ytrain)
    base_r2 = r2_score(ytest, lr_model.predict(xtest))
    print(f"linear regression r²: {base_r2:.4f}")
    print("-" * 30)

    better = 0
    for model in models:
        model.fit(xtrain, ytrain)
        r2 = r2_score(ytest, model.predict(xtest))
        status = "better" if r2 > base_r2 else "worse"
        if r2 > base_r2:
            better += 1
            print(f"{type(model).__name__} outperformed linear regression")
        print(f"{type(model).__name__}: {r2:.4f} ({status})")

    print(f"\n{better}/{len(models)} models beat linear regression")

vs_linear(lr_model, models, xtrain, xtest, ytrain, ytest)
print("=" * 50)
best_model, best_r2 = evaluate_models(models, xtrain, xtest, ytrain, ytest)
models.append(best_model)
print(f"best model: {best_model} with r² = {best_r2:.4f}")

optimized_model, best_params = optimize_best_model(best_model, xtrain, ytrain, xtest, ytest)
print(f"\nfinal optimized model: {optimized_model}")
print("final best parameters:", best_params)

joblib.dump(optimized_model, 'BESTHOUSE.pkl')
joblib.dump(scaler, 'scaler.pkl')
