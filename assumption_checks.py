# assumption_checks.py
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

df = pd.read_csv("aadhar.csv")
X = df[['age_0_5', 'age_5_17', 'age_18_greater']]
y = X.sum(axis=1)  # Example target

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
residuals = model.resid

# Residuals Plot
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residuals Histogram")
plt.savefig("residuals_plot.png")

# Normality QQ-plot
sm.qqplot(residuals, line="45")
plt.title("Residuals Q-Q Plot")
plt.savefig("qq_plot.png")

# Multicollinearity (VIF)
vif_data = []
for i in range(X.shape[1]):
    vif_data.append({
        "variable": X.columns[i],
        "VIF": variance_inflation_factor(X.values, i)
    })
vif_df = pd.DataFrame(vif_data)
vif_df.to_csv("vif_values.csv", index=False)
print(vif_df)
