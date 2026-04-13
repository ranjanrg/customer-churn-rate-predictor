import pandas as pd
from sklearn.model_selection import train_test_split

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
    'MultipleLines',
    'PaymentMethod'
]

df = pd.get_dummies(df, columns=cols)

# print(df.info())

x = df.drop('Churn', axis = 1)
y = df['Churn']

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)