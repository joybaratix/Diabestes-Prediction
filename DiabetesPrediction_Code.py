import pandas as pd
# Load the dataset
df = pd.read_csv("https://github.com/joybaratix/Diabestes-Prediction/blob/main/diabetes_raw_dataset.csv")
df


from sklearn.model_selection import train_test_split
# Split the dataset into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Perform feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
# Train the logistic regression model
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy score and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion matrix:')
print(cm)

import numpy as np
# Create a new data point to predict
new_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Preprocess the new data point
new_data = imputer.transform(new_data)
new_data = scaler.transform(new_data)

# Make a prediction on the new data point
prediction = model.predict(new_data)
print('Prediction:', prediction)


import matplotlib.pyplot as plt
# Pie chart of diabetes and non-diabetes patients
plt.pie(df['Outcome'].value_counts(), labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Diabetic vs Non-Diabetic Patients')
plt.axis('equal')
plt.show()

# Bar chart of average glucose level for diabetic vs non-diabetic patients
avg_glucose = df.groupby('Outcome')['Glucose'].mean()
plt.bar(['Non-Diabetic', 'Diabetic'], avg_glucose)
plt.title('Average Glucose Level for Diabetic vs Non-Diabetic Patients')
plt.xlabel('Patient Outcome')
plt.ylabel('Average Glucose Level')
plt.show()

# Scatter plot of BMI vs age for diabetic and non-diabetic patients
non_diabetic = df[df['Outcome'] == 0]
diabetic = df[df['Outcome'] == 1]
plt.scatter(non_diabetic['Age'], non_diabetic['BMI'], label='Non-Diabetic', alpha=0.5)
plt.scatter(diabetic['Age'], diabetic['BMI'], label='Diabetic', alpha=0.5)
plt.title('BMI vs Age for Diabetic vs Non-Diabetic Patients')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.legend()
plt.show()

