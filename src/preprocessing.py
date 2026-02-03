import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df.dropna(subset=['Life_expectancy']).copy()

def split_data(df_cleaned):
    numerical_features = [
        'Year', 'Adult_Mortality', 'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B',
        'Measles', 'BMI', 'Polio', 'Total_expenditure', 'Diphtheria', 'GDP', 'Population',
        'thinness__1-19_years', 'thinness_5-9_years', 'Income_composition_of_resources', 'Schooling'
    ]
    categorical_features = ['Country', 'Status']
    all_features = df_cleaned.columns.drop(['Life_expectancy', 'under-five_deaths']).tolist()
    X = df_cleaned[all_features]
    y = df_cleaned['Life_expectancy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, numerical_features, categorical_features

