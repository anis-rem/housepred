import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
scaler = StandardScaler()
pd.set_option('display.max_columns', None)
df = pd.read_csv("Housing.csv")
print(df.head())
# Data preprocessing
df.replace('yes', 1, inplace=True)
df.replace('no', 0, inplace=True)
df.replace('furnished', 1, inplace=True)
df.replace('unfurnished', 0, inplace=True)
df.replace('semi-furnished', -1, inplace=True)

# Feature engineering
df["total rooms"] = df["bathrooms"] + df["bedrooms"]
df["room per floor"] = df["total rooms"] / df["stories"].replace(0, 1)

df['yess score'] = (df['mainroad'] + df['guestroom'] + df['basement'] +
                    df['hotwaterheating'] + df['airconditioning'] + df['prefarea'])

print(df.head(1))
print("Missing values:")
print(df.isnull().sum())
# Feature scaling
df["area"] *= 1.3
df["total rooms"] *= 1.2
df["room per floor"] *= 1.1
df["yess score"] *= 1.5

# Prepare data
x = df.drop(columns=["price"])
x_scaled = scaler.fit_transform(x)
y = df["price"]
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2)

print("Testing data:", xtest.shape)
print("Training data:", xtrain.shape)

# Initialize models
LRmodel = LinearRegression()
DsTreemodel = DecisionTreeRegressor(random_state=42)
Rfmodel = RandomForestRegressor(random_state=42)
Ridgemodel = Ridge(tol=1e-3)  # Less strict convergence
Lassomodel = Lasso(tol=1e-3)
Elastic = ElasticNet(tol=1e-3)


models = [Ridgemodel, Lassomodel, Elastic, LRmodel, DsTreemodel, Rfmodel]
modelsnolinearregress = [Ridgemodel, Lassomodel, Elastic, LRmodel, DsTreemodel, Rfmodel]

def get_param_grid(model):
    model_name = type(model).__name__

    if model_name == 'RandomForestRegressor':
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

    elif model_name == 'DecisionTreeRegressor':
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

    elif model_name == 'Ridge':
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "fit_intercept": [True, False],
            "solver": ["auto", "svd", "cholesky", "sparse_cg", "sag", "saga"],
            "max_iter": [1000, 2000, 10000, 20000]

        }

    elif model_name == 'Lasso':
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True, False],
            "precompute": [True, False],
            "max_iter": [1000, 2000, 5000, 10000],
            "tol": [1e-4, 1e-3, 1e-2],
            "positive": [True, False],
            "selection": ["cyclic", "random"]
        }

    elif model_name == 'ElasticNet':
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

    elif model_name == 'LinearRegression':
        return {
            "fit_intercept": [True, False],
            "positive": [True, False]
        }

    else:
        return {}


def evaluate_models(models, xtrain, xtest, ytrain, ytest):
    """
    Evaluate multiple models and return the best one
    """
    best_model = None
    best_r2 = float('-inf')

    for model in models:
        print(f"\n{type(model).__name__} results:")
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        mae = mean_absolute_error(ytest, ypred)
        mse = mean_squared_error(ytest, ypred)
        r2 = r2_score(ytest, ypred)

        print("MAE:", round(mae, 2))
        print("MSE:", round(mse, 2))
        print("R² Score:", round(r2, 4))

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    return best_model, best_r2


def optimize_best_model(best_model, xtrain, ytrain, xtest, ytest):
    model_name = type(best_model).__name__
    print(f"\nOptimizing {model_name}...")
    param_grid = get_param_grid(best_model)
    random_search = RandomizedSearchCV(
        estimator=type(best_model)(),
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        verbose=1,
        n_jobs=-1,
        scoring='r2',
        random_state=42,
    )
    random_search.fit(xtrain, ytrain)
    optimized_model = random_search.best_estimator_
    ypred_optimized = optimized_model.predict(xtest)
    r2_optimized = r2_score(ytest, ypred_optimized)

    print(f"\nOptimized {model_name} Results:")
    print("Best params:", random_search.best_params_)
    print("Test R² Score:", round(r2_optimized, 4))

    return optimized_model, random_search.best_params_
def Vslinearregression(LRmodel,models, xtrain, xtest, ytrain, ytest):
    LRmodel.fit(xtrain, ytrain)
    lr_r2 = r2_score(ytest, LRmodel.predict(xtest))
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print("-" * 30)
    better_count = 0
    for model in models:
        model.fit(xtrain, ytrain)
        r2 = r2_score(ytest, model.predict(xtest))
        status = "BETTER" if r2 > lr_r2 else "WORSE"
        if r2 > lr_r2:
            better_count += 1
            print(f"{type(model).__name__} outperformed linear regession")
        print(f"{type(model).__name__}: {r2:.4f} ({status})")

    print(f"\n{better_count}/{len(models)} models beat LinearRegression")

results = Vslinearregression(LRmodel,models, xtrain, xtest, ytrain, ytest)
print("=" * 50)
best_model, best_r2 = evaluate_models(models, xtrain, xtest, ytrain, ytest)
models.append(best_model)
print(f"\n{'=' * 50}")
print(f"BEST MODEL: {(best_model) }with R² = {best_r2:.4f}")
print(f"{'=' * 50}")
optimized_model, best_params = optimize_best_model(best_model, xtrain, ytrain, xtest, ytest)
print(f"\nFinal optimized model: {(optimized_model)}")
print("Final best parameters:", best_params)
joblib.dump(optimized_model, 'BESTHOUSE.pkl')
joblib.dump(scaler, 'scaler.pkl')
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)  # only numeric columns
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Number of Houses')
plt.grid(True)
plt.show()
df = df[df['price'] < 15_000_000]

# Show how many houses are under 2 million
under_2m_count = df[df['price'] < 2_000_000].shape[0]
print(f"Houses under 2M: {under_2m_count}")

# Plot histogram
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=40, kde=True, color='skyblue', edgecolor='black')

# Title and labels
plt.title('Distribution of House Prices', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Format x-axis to show labels in millions (e.g., 1M, 2M, ...)
formatter = ticker.FuncFormatter(lambda x, _: f'{int(x / 1e6)}M')
plt.gca().xaxis.set_major_formatter(formatter)

# Set x-axis to start at 0 for full range visibility
plt.xlim(0, df['price'].max())

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['area'], bins=40, kde=True, color='lightgreen', edgecolor='black')

plt.title('Distribution of House Area', fontsize=16)
plt.xlabel('Area (sq ft)', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

df_plot = df.copy()
df_plot['furnishingstatus'] = df_plot['furnishingstatus'].map({
    0: 'Unfurnished',
   -1: 'Semi-furnished',
    1: 'Furnished'
})

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_plot, x='furnishingstatus', y='price', palette='Set2')

plt.title('House Prices by Furnishing Status', fontsize=16)
plt.xlabel('Furnishing Status', fontsize=12)
plt.ylabel('Price', fontsize=12)

# Format y-axis to millions
formatter = ticker.FuncFormatter(lambda x, _: f'{int(x / 1e6)}M')
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(df['area'], df['price'], c='blue', edgecolors='black', alpha=0.7)
plt.title('Scatterplot of House Price vs Area', fontsize=14)
plt.xlabel('Area (sq ft)', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True)
plt.show()