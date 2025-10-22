import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# === Step 1: Load dataset ===
df = pd.read_csv("archive/data.csv")
df = df.dropna().drop_duplicates()

# === Step 2: Encode categorical variables ===
categorical_cols = [
    'Gender', 'Department', 'Extracurricular_Activities',
    'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level'
]

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# === Step 3: Map Grades to "F" vs "Not F" ===
df['Grade'] = df['Grade'].astype(str).str.upper()  # ensure uppercase like 'A', 'B', 'F'
df = df[df['Grade'].isin(['A', 'B', 'C', 'D', 'F'])]  # keep standard letters
df['Is_F'] = np.where(df['Grade'] == 'F', 1, -1)  # 1 = F, -1 = Not F

# === Step 4: Feature selection ===
features = [
    'Gender', 'Age', 'Department', 'Study_Hours_per_Week',
    'Extracurricular_Activities', 'Sleep_Hours_per_Night',
    'Attendance (%)', 'Stress_Level (1-10)',
    'Parent_Education_Level', 'Family_Income_Level', 'Internet_Access_at_Home'
]
X = df[features].values
y = df['Is_F'].values

# === Step 5: Scaling + PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.9, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 6: Custom Linear SVM ===
class LinearSVM:
    def __init__(self, learning_rate=0.0005, lambda_param=0.01, n_iters=2000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    db = y[idx]
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# === Step 7: Train + Evaluate ===
svm = LinearSVM(learning_rate=0.0005, lambda_param=0.01, n_iters=2000)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🎯 Custom Linear SVM Accuracy (F vs Not F): {acc * 100:.2f}%")

# Optional: print balance of F vs Not F
unique, counts = np.unique(y_test, return_counts=True)
print(f"Class balance in test set: {dict(zip(unique, counts))}")

# === Step 8: Visualization (2D PCA plot) ===
import matplotlib.pyplot as plt

## 1️⃣ Distribution of numeric features by F/Not F
numeric_features = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)', 'Age']
import seaborn as sns

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(data=df, x=feature, hue="Is_F", fill=True, common_norm=False, palette="coolwarm")
    plt.title(f"{feature} Distribution")
plt.tight_layout()
plt.show()

## 2️⃣ Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[features + ['Is_F']].corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap (Including F/Not F)")
plt.show()

## 3️⃣ PCA 2D scatter (visual separation)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Is_F", palette="coolwarm", alpha=0.7)
plt.title("PCA Projection: F (1) vs Not F (0)")
plt.show()
