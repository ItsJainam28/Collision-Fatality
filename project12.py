# -*- coding: utf-8 -*-
"""
Created on March 20 2024

@title: Traffic Collision - Supervised Learning(COMP247)
@author: Group 7 - Section 001
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# Ignore warnings
warnings.simplefilter("ignore")

# =============================================================================
# Phase 1: Data exploration, Cleaning and Modeling
# =============================================================================

# Load dataset
ksi_df7 = pd.read_csv("KSI.csv")
pd.set_option("display.max_columns", None)

# Display first few rows of the dataset
ksi_df7.head()

# Display column names
ksi_df7.columns

# Display shape of the dataset
ksi_df7.shape

# Display data types of columns
ksi_df7.dtypes

# Check the missing values
ksi_df7.isnull().values.any()

# Display dataset information
ksi_df7.info()

ksi_df7.isnull().sum()

# Check the statistics
ksi_df7.describe().T

# Replace Null values with NaN
ksi_df7.replace(" ", np.nan)
ksi_df7.replace("", np.nan)

def remove_cols(df: pd.DataFrame, column_names: list):
    """
    Remove specified columns from the DataFrame.
    
    Parameters:
         (DataFrame): The DataFrame from which columns are to be removed.
        column_names (list): List of column names to be removed.
    """
    df.drop(column_names, axis=1, inplace=True)

# Remove unnecessary columns: INDEX_, ObjectId, X, Y
remove_cols(ksi_df7, ["INDEX_", "ObjectId", "X", "Y"])

# Date decomposition
ksi_df7["DATE"] = pd.to_datetime(ksi_df7["DATE"], format="%Y/%m/%d %H:%M:%S%z")
ksi_df7.insert(1, "MONTH", ksi_df7["DATE"].dt.month)
ksi_df7.insert(2, "DAY", ksi_df7["DATE"].dt.day)
ksi_df7.drop(["DATE", "TIME"], axis=1, inplace=True)

# Function to get range of numerical columns
def get_range(x):
    return pd.Series(index=["min", "max"], data=[x.min(), x.max()])

# Display range for each numerical column
ksi_df7.select_dtypes(include=np.number).apply(get_range)

# Display descriptive statistics of dataset
ksi_df7.describe()

# Handle boolean columns
bool_cols = [
    "PEDESTRIAN",
    "CYCLIST",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "TRUCK",
    "TRSN_CITY_VEH",
    "EMERG_VEH",
    "PASSENGER",
    "SPEEDING",
    "AG_DRIV",
    "REDLIGHT",
    "ALCOHOL",
    "DISABILITY",
]

# Replace Yes with 1 and NaN with 0
ksi_df7[bool_cols] = ksi_df7[bool_cols].replace({"Yes": 1, np.nan: 0})


# Handle mapping for categorical columns
categorical_mapping = {
    "ROAD_CLASS": {
        "Major Arterial": "Arterial",
        "Local": "Local",
        "Minor Arterial": "Arterial",
        "Collector": "Collector",
        "Other": "Other",
        "Pending": "Other",
        "Laneway": "Other",
        "Expressway": "Expressway",
        "Expressway Ramp": "Expressway",
        "Major Arterial Ramp": "Arterial",
    },
    "LOCCOORD": {
        "Intersection": "Intersection",
        "Mid-Block": "Non Intersection",
        "Exit Ramp Westbound": "Non Intersection",
        "Exit Ramp Southbound": "Non Intersection",
        "Mid-Block (Abnormal)": "Non Intersection",
        "Entrance Ramp Westbound": "Non Intersection",
        "Park, Private Property, Public Lane": "Non Intersection",
    },
    "ACCLOC": {
        "At Intersection": "Intersection",
        "Intersection Related": "Intersection",
        "Non Intersection": "Non Intersection",
        "At/Near Private Drive": "Non Intersection",
        "Underpass or Tunnel": "Non Intersection",
        "Private Driveway": "Non Intersection",
        "Overpass or Bridge": "Non Intersection",
        "Trail": "Non Intersection",
        "Laneway": "Non Intersection",
    },
    "TRAFFCTL": {
        "No Control": "No Control",
        "Stop Sign": "Control",
        "Traffic Signal": "Control",
        "Pedestrian Crossover": "Control",
        "Traffic Controller": "Control",
        "Yield Sign": "Control",
        "School Guard": "Control",
        "Traffic Gate": "Control",
        "Police Control": "Control",
        "Streetcar (Stop for)": "Control",
    },
    "DRIVACT": {
        "Driving Properly": "Normal",
        "Lost control": "Not Normal",
        "Failed to Yield Right of Way": "Not Normal",
        "Improper Passing": "Not Normal",
        "Improper Turn": "Not Normal",
        "Exceeding Speed Limit": "Not Normal",
        "Disobeyed Traffic Control": "Not Normal",
        "Following too Close": "Not Normal",
        "Other": "Not Normal",
        "Improper Lane Change": "Not Normal",
        "Wrong Way on One Way Road": "Not Normal",
        "Speed too Fast For Condition": "Not Normal",
        "Speed too Slow": "Not Normal",
    },
    "DRIVCOND": {
        "Normal": "Normal",
        "Ability Impaired, Alcohol Over .08": "Not Normal",
        "Unknown": "Not Normal",
        "Inattentive": "Not Normal",
        "Had Been Drinking": "Not Normal",
        "Medical or Physical Disability": "Not Normal",
        "Ability Impaired, Alcohol": "Not Normal",
        "Fatigue": "Not Normal",
        "Other": "Not Normal",
        "Ability Impaired, Drugs": "Not Normal",
    },
    "VISIBILITY": {
        "Clear": "Clear",
        "Rain": "Not Clear",
        "Other": "Not Clear",
        "Snow": "Not Clear",
        "Strong wind": "Not Clear",
        "Fog, Mist, Smoke, Dust": "Not Clear",
        "Drifting Snow": "Not Clear",
        "Freezing Rain": "Not Clear",
    },
    "LIGHT": {
        "Daylight": "Daylight",
        "Dark": "Dark",
        "Dusk": "Dark",
        "Dark, artificial": "Dark",
        "Dusk, artificial": "Dark",
        "Dawn, artificial": "Daylight",
        "Dawn": "Daylight",
        "Daylight, artificial": "Daylight",
        "Other": "Dark",
    },
    "RDSFCOND": {
        "Dry": "Dry",
        "Wet": "Wet",
        "Other": "Wet",
        "Slush": "Wet",
        "Loose Snow": "Dry",
        "Ice": "Wet",
        "Packed Snow": "Dry",
        "Spilled liquid": "Wet",
        "Loose Sand or Gravel": "Dry",
    }
}

# Replace categorical values in DataFrame columns based on provided mappings
for column, mapping in categorical_mapping.items():
    ksi_df7[column] = ksi_df7[column].replace(mapping)

# Replace specific values in "DISTRICT" column
ksi_df7["DISTRICT"].replace(
    "Toronto East York", "Toronto and East York", inplace=True)

# Replace specific values in "ACCLASS" column
ksi_df7["ACCLASS"].replace(
    ["Property Damage Only", "Non-Fatal Injury"], "Non-Fatal", inplace=True
)

# Display count of missing values in DataFrame
ksi_df7.isnull().sum().sort_values()


# Fill NaN in "LOCCOORD" column with values from "ACCLOC" where NaN exists
ksi_df7["LOCCOORD"].fillna(ksi_df7["ACCLOC"], inplace=True)
# Check if there are still columns with null values
ksi_df7.columns[ksi_df7.isin([np.nan]).any()]

# Drop columns with high percentage of missing values
cols_with_nulls = [
    "ACCNUM",
    "STREET1",
    "STREET2",
    "OFFSET",
    "WARDNUM",
    "DIVISION",
    "ACCLOC",
    "TRAFFCTL",
    "IMPACTYPE",
    "INVTYPE",
    "INITDIR",
    "VEHTYPE",
    "MANOEUVER",
    "DRIVCOND",
    "PEDTYPE",
    "PEDACT",
    "PEDCOND",
    "CYCLISTYPE",
    "CYCACT",
    "CYCCOND",
    "FATAL_NO",
    "NEIGHBOURHOOD_158",
    "NEIGHBOURHOOD_140",
]

# Drop columns with high null value and low correlation
ksi_df7.drop(cols_with_nulls, axis=1, inplace=True)

# Correlation Analysis after dropping unrelated columns
ksi_df7_copy = ksi_df7.copy()
ksi_df7_copy["ACCLASS"] = ksi_df7_copy["ACCLASS"].map({"Fatal": 1, 
                                                       "Non-Fatal": 0})

correlation_matrix = ksi_df7_copy.select_dtypes(include=[np.number]).corr()

# To visualize the correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Dataset")
plt.show()

# Import necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

def plot_categorical_count(data, x, hue, height=8.27, aspect=None):
    """
    Plot categorical count based on specified variables.

    Parameters:
        data (DataFrame): The DataFrame containing the data.
        x (str): The variable to be plotted on the x-axis.
        hue (str): The variable to be used for coloring the plot.
        height (float): The height of the plot in inches. Default is 8.27.
        aspect (float): The aspect ratio of the plot. Default is None.
    """
    sns.catplot(x=x, kind="count", data=data, hue=hue, height=height, aspect=aspect)

# Yearly Trend of Accidents
plt.figure(figsize=(12, 6))
sns.countplot(x="YEAR", data=ksi_df7, palette="viridis")
plt.title("Yearly Trend of Accidents")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()

# Plot number accidents by year for each class
plot_categorical_count(ksi_df7, "YEAR", "ACCLASS", aspect=11.7 / 8.27)

# Plotting the histogram for "ACCLASS"
plt.figure(figsize=(8, 6))
sns.histplot(ksi_df7["ACCLASS"], bins=2, kde=False)
plt.title("Histogram of ACCCLASS")
plt.xlabel("Accident Classification")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Fatal", "Non-Fatal"])
plt.show()

# Create histogram plots for the dataset
ksi_df7.hist(figsize=(20, 15))

# Plot district wise accidents count
plot_categorical_count(ksi_df7, "DISTRICT", "ACCLASS", aspect=11.7 / 8.27)

# Plot road type wise accidents count
plot_categorical_count(ksi_df7, "ROAD_CLASS", "ACCLASS", aspect=15.7 / 8.27)

# Plot location wise accidents count
plot_categorical_count(ksi_df7, "LOCCOORD", "ACCLASS", aspect=17.7 / 8.27)

# Plot visibility wise
plot_categorical_count(ksi_df7, "VISIBILITY", "ACCLASS", aspect=15.7 / 8.27)

# Plot rdsfcond wise
plot_categorical_count(ksi_df7, "RDSFCOND", "ACCLASS", aspect=15.7 / 8.27)

# Plot driver action wise accidents
plot_categorical_count(ksi_df7, "DRIVACT", "ACCLASS", aspect=25.7 / 8.27)

# Print percentage of missing values for each feature
ksi_df7.isna().sum() / len(ksi_df7) * 100

# Convert object type columns to categorical
categorical_cols = ksi_df7.select_dtypes(include="object").columns
ksi_df7[categorical_cols] = ksi_df7[categorical_cols].astype("category")

# Create copy for data manipulation
pipeline_df7 = ksi_df7.copy()

# Categorical Features
df_categorical = pipeline_df7.select_dtypes(
    include=["category"]).drop("ACCLASS", axis=1)

# Numeric Features
df_numeric = pipeline_df7[["MONTH", "DAY", "YEAR", "LATITUDE", "LONGITUDE"]]

# ColumnTransformer
numeric_features = df_numeric.columns
categorical_features = df_categorical.columns

# Convert the dependent variable to numeric
classification = pd.get_dummies(pipeline_df7["ACCLASS"])
pipeline_df7 = pd.concat([pipeline_df7, classification], axis=1)
pipeline_df7.drop("ACCLASS", axis=1, inplace=True)
pipeline_df7.drop("Non-Fatal", axis=1, inplace=True)

sns.heatmap(
    pipeline_df7.select_dtypes(include=["number"]).corr(method="pearson"), 
    cmap="viridis"
)
#plt.show()

# Define preprocessing steps
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

preprocessor.transformers

# Features and Target
features = pipeline_df7.drop("Fatal", axis=1)
target = pipeline_df7["Fatal"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.20, random_state=15, stratify=target
)

# Define logistic regression model
log_reg_model7 = Pipeline(
    steps=[("preprocessor", preprocessor), 
           ("classifier", LogisticRegression())]
)

# Fit the Logistic Regression Model
log_reg_model7.fit(X_train, y_train)

# Make predictions on testing data
y_pred_log = log_reg_model7.predict(X_test)

# Evaluate accuracy of Logistic Regression Model
accuracy_logistic = accuracy_score(y_test, y_pred_log)
print("Accuracy of Logistic regression model:", accuracy_logistic)

target.value_counts()

# Define SVC model
pipeline_svc = Pipeline(
    steps=[("preprocessor", preprocessor), 
           ("classifier", SVC(kernel='linear', 
            class_weight='balanced',
            probability=True))])

# Fit the SVC model
pipeline_svc.fit(X_train, y_train)

# Make predictions on testing data
y_pred_svc = pipeline_svc.predict(X_test)

# Evaluate accuracy of SVC model
accuracy_SVC = accuracy_score(y_test, y_pred_svc)
print("Accuracy of SVC model:", accuracy_SVC)

prob_y_svc = pipeline_svc.predict_proba(features)
prob_y_svc = [p[1] for p in prob_y_svc]

from sklearn.metrics import roc_auc_score
# AUROC represents the likelihood 
# of model distinguishing observations from two classes.
print(roc_auc_score(target, prob_y_svc) )
# =============================================================================
# preprocessed_names = preprocessor.get_feature_names_out()
# transformed_data = preprocessor.transform(pipeline_df7)
# 
# # Convert the transformed data back to a DataFrame
# transformed_df = pd.DataFrame(transformed_data, columns=preprocessed_names)
# 
# # Save Transformed dataset to CSV
# transformed_df.to_csv("data\ksi_data_cleaned.csv", index=False)
# =============================================================================
