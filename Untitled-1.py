# %%
import pandas as pd

# %%
df = pd.read_csv('KSI.csv')

# %%
# How big is the dataset?
print(df.shape)

# %%
df.sample(5)

# %%
df.info()


# %%
df.isnull().sum()

# %%
def is_fatal(injury):
    if injury == 'Fatal':
        return 1
    else:
        return 0

df['is_fatal'] = df['INJURY'].apply(is_fatal)
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_df.corr()['is_fatal']
print(correlation)

# %%
categorical_columns = df.select_dtypes(include=['object']).columns
unique_counts = df[categorical_columns].nunique()
print(unique_counts)
filtered_columns = unique_counts[unique_counts <= 160].index
columns_to_drop = ['HOOD_140', 'HOOD_158', 'NEIGHBOURHOOD_140', 'DIVISION']
filtered_columns = filtered_columns.drop(columns_to_drop, errors='ignore')
print(filtered_columns)
null_counts = df[filtered_columns].isnull().sum()
print(null_counts)
df_encoded = pd.get_dummies(df[filtered_columns])
# Add 'IS_FATAL' column to encoded DataFrame
df_encoded['IS_FATAL'] = df['is_fatal']
# Calculate correlation
correlation = df_encoded.corr()['IS_FATAL']
pd.set_option('display.max_rows', None)
print(correlation)

# %%
df = df_encoded
print(df.shape)
print(df.columns)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming 'IS_FATAL' is your target variable
X = df.drop('IS_FATAL', axis=1)
y = df['IS_FATAL']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


