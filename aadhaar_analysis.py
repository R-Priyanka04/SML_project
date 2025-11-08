# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Load Data
df = pd.read_csv("aadhar.csv")
print(df.head())

# 3. Exploratory Data Analysis
df.info()
df.describe()
sns.histplot(df['age_0_5'])
plt.show()
# Additional EDA plots...

# 4. Prepare Data for Regression
X = df[['age_0_5', 'age_5_17', 'age_18_greater']]  # adjust as needed
y = X.sum(axis=1)  # example target, modify per your question

# 5. Model Fitting
model = LinearRegression()
model.fit(X, y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 6. Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
print("Cross-validated R^2:", scores.mean())

# 7. Save Outputs
df['predicted'] = model.predict(X)
df.to_csv("output_predictions.csv", index=False)

# 8. Visualize Results
plt.scatter(y, df['predicted'])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted")
plt.show()
