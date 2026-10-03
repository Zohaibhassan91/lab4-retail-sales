import numpy as np
import pandas as pd
from scipy import stats

# Input data
revenue_data = [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
df = pd.DataFrame({
    'Day': range(1, 11),
    'Revenue': revenue_data
})

# (a) Report missing values and fill with median
missing_count = df['Revenue'].isnull().sum()
print(f"Missing values in Revenue: {missing_count}\n")

median_revenue = df['Revenue'].median()
df['Revenue'] = df['Revenue'].fillna(median_revenue)

# (b) Min-Max Normalization and Z-score Standardization
revenue_min = df['Revenue'].min()
revenue_max = df['Revenue'].max()
df['Revenue_MinMax'] = (df['Revenue'] - revenue_min) / (revenue_max - revenue_min)

df['Revenue_Zscore'] = stats.zscore(df['Revenue'])

# (c) Detect Outliers using IQR Method
q1 = df['Revenue'].quantile(0.25)
q3 = df['Revenue'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df['Outlier'] = (df['Revenue'] < lower_bound) | (df['Revenue'] > upper_bound)
outlier_days = df[df['Outlier']]['Day'].tolist()

# (d) Print Final Table and Results
print("Final Data Table:")
print(df.to_string(index=False))
print("-" * 50)
print(f"IQR outliers detected on day(s): {outlier_days}")