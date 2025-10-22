# Exploratory Data Analysis
# working on the distributions of key features as histograms

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('student_performance_dataset.csv')

gender = df['Gender']

age = df['Age']

department = df['Department']

attendance = df['attendance']

plt.hist(gender, bins=30)


