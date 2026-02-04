# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, drop unnecessary columns, and encode categorical variables.

2. Define the features (X) and target variable (y).

3. Split the data into training and testing sets.

4. Train the logistic regression model, make predictions, and evaluate using accuracy and other .
   

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: pulla kavya
RegisterNumber: 212225240110
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
dataset = pd.read_csv("C:/Users/acer/Downloads/Placement_Data.csv")
dataset.head()
df = dataset.copy()
df = df.drop(columns=['sl_no', 'salary'])

df.head()
print("Missing Values in Dataset:\n")
print(df.isnull().sum())

print("\nTotal Duplicate Rows:", df.duplicated().sum())
encoder = LabelEncoder()

df['gender'] = encoder.fit_transform(df['gender'])
df['ssc_b'] = encoder.fit_transform(df['ssc_b'])
df['hsc_b'] = encoder.fit_transform(df['hsc_b'])
df['hsc_s'] = encoder.fit_transform(df['hsc_s'])
df['degree_t'] = encoder.fit_transform(df['degree_t'])
df['workex'] = encoder.fit_transform(df['workex'])
df['specialisation'] = encoder.fit_transform(df['specialisation'])
df['status'] = encoder.fit_transform(df['status'])

df.head()
X = df.drop('status', axis=1)
y = df['status']

print("Input Features Shape:", X.shape)
print("Output Target Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=10
)
log_reg = LogisticRegression(
    solver='liblinear',
    max_iter=1000
)

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print("Predicted Values:\n", y_pred)
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", acc)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
cmd = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Placed', 'Placed']
)

cmd.plot()
plt.title("Confusion Matrix – Logistic Regression")
plt.show()

```

## Output:
<img width="726" height="434" alt="image" src="https://github.com/user-attachments/assets/4cf00b17-5545-4df0-acf3-6134fd218350" />
<img width="959" height="659" alt="image" src="https://github.com/user-attachments/assets/597a1644-28d7-4150-9d5d-388c9f6596c3" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
