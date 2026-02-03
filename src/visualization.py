import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predictions(y_test, y_pred):
    min_y = min(y_test.min(), y_pred.min())
    max_y = max(y_test.max(), y_pred.max())
    bins = np.linspace(min_y, max_y, 30)
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=bins, alpha=0.6, label="Predicted values", color='blue', edgecolor='black')
    plt.hist(y_test, bins=bins, alpha=0.6, label="True values", color='red', edgecolor='black')
    plt.xlabel("Lifetime")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

def plot_feature_scatter(X_train, y_train):
    target = 'Life_expectancy'
    not_include = ['Country', 'Year']
    final_plot_data = X_train.copy()
    final_plot_data[target] = y_train
    features = [col for col in X_train.columns if col not in not_include]
    n_cols = 2
    n_rows = (len(features) + n_cols - 1) // n_cols
    plt.figure(figsize=(12, 5 * n_rows))
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.scatterplot(data=final_plot_data, x=feature, y=target, alpha=0.6, color='royalblue')
        if feature == 'Population':
            plt.xscale('log')
        plt.title(feature.replace("_", " "))
        plt.ylabel("Life Expectancy")
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(X_train):
    not_include = ["Year", "Status", "infant_deaths", "percentage_expenditure", "Measles", 'under-five_deaths', 'HIV/AIDS', "GDP", "Population", "Country"]
    features = [i for i in X_train.columns if i not in not_include]
    n_cols = 2
    n_rows = (len(features) + n_cols - 1) // n_cols
    plt.figure(figsize=(12, 5 * n_rows))
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(X_train[feature], kde=True, color='seagreen')
        plt.title(f"Distribution of {feature.replace('_', ' ')}")
    plt.tight_layout()
    plt.show()
