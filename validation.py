# model_validation.py
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv("aadhar.csv")
X = df[['age_0_5', 'age_5_17', 'age_18_greater']]
y = X.sum(axis=1)

model = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')

with open("cv_results.txt", "w") as f:
    f.write(f"Cross-validated R^2 scores: {scores}\n")
    f.write(f"Mean R^2: {scores.mean()}\n")
