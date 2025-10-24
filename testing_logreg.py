
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
# dataset faulty bcs too big... 
# Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection 

df = pd.read_csv('fraudTest.csv')

# Encode Grade (A,B,C,D,F) → numeric labels (e.g., A=0, B=1, etc.)
# le = LabelEncoder()
# df['Grade_encoded'] = le.fit_transform(df['Grade'])
# print(dict(zip(le.classes_, le.transform(le.classes_)))) 

categorical_X = df[['category', 'gender', 'job']]
numeric_X= df[['amt', 'lat', 'long', 'merch_lat', 'merch_long']]
y = df['is_fraud']

# Standardize numeric features for interpretability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_X)

# Train multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_scaled, y)

coefficients = pd.DataFrame(model.coef_, columns=numeric_X.columns)
print("Feature coefficients by grade level:\n")
print(coefficients)

plt.figure(figsize=(10, 6))
sns.heatmap(coefficients.T, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Influence on Each Grade Level')
plt.xlabel('Grade')
plt.ylabel('Feature')
plt.show()