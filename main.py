from src.preprocessing import clean_data, get_preprocessor
from src.visualize import plot_distributions
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, EXCLUDE_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

#Տվյալների նախապատրաստում
df = clean_data("data/Life Expectancy Data.csv")
X = df.drop([TARGET, 'under-five_deaths'], axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Pipeline-ի ստեղծում
preprocessor = get_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

#Մարզում և կանխատեսում
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# RMSE հաշվարկ
mse = mean_squared_error(y_test, y_pred)
print(f"RMSE: {mse ** 0.5}")

# Վիզուալիզացիա (ըստ քո logic-ի)
features_to_plot = [i for i in X_train.columns if i not in EXCLUDE_FEATURES]