import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression  # Example model

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
sigma_df = pd.read_csv('KSI.csv')

# Custom transformer for age categorization
class AgeCategorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        def age_category(age_range):
            if age_range in ['0 to 4', '5 to 9', '10 to 14']:
                return 'kid'
            elif age_range == '15 to 19':
                return 'teenager'
            elif age_range in ['20 to 24', '25 to 29']:
                return 'youth'
            elif age_range in ['30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64']:
                return 'adult'
            elif age_range in ['65 to 69', '70 to 74', '75 to 79', '80 to 84', '85 to 89', '90 to 94', 'Over 95']:
                return 'old'
            else:
                return 'unknown'
        return X.applymap(age_category)

# Define the preprocessing for categorical columns
categorical_preprocessing = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Define the preprocessing for age column
age_preprocessing = Pipeline(steps=[
    ('age_categorize', AgeCategorizer()),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Column Transformer to apply transformations to different columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_preprocessing, ['ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INJURY', 'INITDIR', 'DRIVACT', 'PEDTYPE', 'CYCACT', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL']),
    ('age', age_preprocessing, ['INVAGE'])
])

# Define the pipeline with preprocessing and a simple classifier for demonstration
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)  # Example model
])

# Split the dataset (assuming 'ACCLASS' is the target)
X = sigma_df.drop('ACCLASS', axis=1)
y = sigma_df['ACCLASS'].replace({'Non-Fatal': 0, 'Fatal': 1})  # Encoding the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)
print(X_train.head())
print(y_train.head())