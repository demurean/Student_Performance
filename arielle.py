# Exploratory Data Analysis
# working on the distributions of key features as histograms

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('student_performance_dataset.csv')

#---------------------------------------------------------------
# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
# df[numeric_cols].hist(bins=20, figsize=(12, 10), edgecolor='black')
# plt.suptitle("Distribution of Numeric Features", fontsize=16)
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Age', y='Grade', data=df, alpha=0.7)
# plt.title('Age vs Grade')
# plt.xlabel('Age')
# plt.ylabel('Grade')
# plt.show()

# plt.figure(figsize=(8, 5))
# sns.histplot(df['Grade'], bins=20, kde=True, color='orange')
# plt.title('Grade Distribution')
# plt.xlabel('Grade')
# plt.ylabel('Frequency')
# plt.show()

# df.plot(x='Age', y='Grade', kind='hist') 
# ?
#--------------------------------------------------------------------

grades = df['Grade']

age = df['Age']
plt.hist(age, bins=30)
plt.xlabel('age')
plt.ylabel('Frequency')
plt.title('age distribution')
plt.show()  

# do a line chart
attendance = df['Attendance (%)']
plt.hist(attendance, bins=30)
plt.xlabel('attendance')
plt.ylabel('Frequency')
plt.title('attendance distribution')
plt.show()

study_hours = df['Study_Hours_per_Week']
plt.hist(study_hours, bins=30)
plt.xlabel('study_hours')
plt.ylabel('Frequency')
plt.title('study_hours distribution')
plt.show()

stress_level = df['Stress_Level (1-10)']
plt.hist(stress_level, bins=30)
plt.xlabel('stress_level')
plt.ylabel('Frequency')
plt.title('stress_level distribution')
plt.show()

sleep_hours = df['Sleep_Hours_per_Night']
plt.hist(sleep_hours, bins=30)
plt.xlabel('sleep_hours')
plt.ylabel('Frequency')
plt.title('sleep_hours distribution')
plt.show()

