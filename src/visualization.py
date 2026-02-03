import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, features, n_rows, n_cols):
    plt.figure(figsize=(12, 5 * n_rows))
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[feature], kde=True, color='seagreen')
        plt.title(f"Distribution of {feature.replace('_', ' ')}")
    plt.tight_layout()
    plt.show()