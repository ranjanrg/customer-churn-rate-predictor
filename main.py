import pandas as pd

df = pd.read_csv('train.csv')

df = df.drop('customerID', axis=1)

df = df[df['TotalCharges'] != " "]

# print((df['TotalCharges'] == " ").sum())

df['TotalCharges'] = df['TotalCharges'].astype(float)

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})

df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})

df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})

df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})

df = pd.get_dummies(df, columns=['InternetService'])

df = pd.get_dummies(df, columns=['Contract'])

cols = [
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'MultipleLines'
]

df = pd.get_dummies(df, columns=cols)

print(df.head())

print(df.info())