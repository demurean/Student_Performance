
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv('student_performance_dataset.csv')

# Encode Grade (A,B,C,D,F) → numeric labels (e.g., A=0, B=1, etc.)
le = LabelEncoder()
df['Grade_encoded'] = le.fit_transform(df['Grade'])
# print(dict(zip(le.classes_, le.transform(le.classes_)))) 

X = df[['Age', 'Attendance (%)', 'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']]
y = df['Grade_encoded']

# Standardize numeric features for interpretability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_scaled, y)

coefficients = pd.DataFrame(model.coef_, columns=X.columns, index=le.classes_)
print("Feature coefficients by grade level:\n")
print(coefficients)

plt.figure(figsize=(10, 6))
sns.heatmap(coefficients.T, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Influence on Each Grade Level')
plt.xlabel('Grade')
plt.ylabel('Feature')
plt.show()