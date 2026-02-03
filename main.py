from src.preprocessing import load_data, split_data
from src.visualization import plot_predictions, plot_feature_scatter, plot_feature_distributions
from src.config import get_preprocessor

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load and split data
df_cleaned = load_data("data/Life Expectancy Data.csv")
X_train, X_test, y_train, y_test, numerical_features, categorical_features = split_data(df_cleaned)

# Build pipeline
preprocessor = get_preprocessor(numerical_features, categorical_features)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_pred_tr = pipeline.predict(X_train)

# Evaluate
rmse = mean_squared_error(y_test, y_pred) ** 0.5
rmse_tr = mean_squared_error(y_train, y_pred_tr) ** 0.5
print(f"Train RMSE: {rmse_tr:.2f}, Test RMSE: {rmse:.2f}")

# Visualizations
plot_predictions(y_test, y_pred)
plot_feature_scatter(X_train, y_train)
plot_feature_distributions(X_train)
