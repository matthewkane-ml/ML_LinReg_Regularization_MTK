# ============================================================
# Linear Regression (Regularization) Project
#
# Problem Statement:
# Sociodemographic and health resource data has been collected at
# the county level across the United States (2018-2019). The goal
# is to determine whether there is a meaningful relationship between
# sociodemographic factors (poverty, education, age distribution,
# race, employment) and health outcomes.
#
# Target variable: Heart disease_number
# The total number of people per county with heart disease. This is
# a raw count target, so county population (TOT_POP) will be an
# important feature because larger counties naturally have more cases.
#
# This is a regression problem. We prepare clean train/test data
# for a regularized Linear Regression model (Lasso).
# ============================================================

# ------------------------------------------------------------
# EDA Pipeline
# 1. High level view of the data - shape, info, identify column types
# 2. Data cleaning - drop redundant/leaky/ID columns, handle nulls and duplicates
# 3. Descriptive statistics
# 4. Univariate analysis - distributions of key features and target
# 5. Verbal analysis of univariate graphs
# 6. Multivariate analysis - correlation heatmap, scatter plots vs target
# 7. Verbal analysis of multivariate graphs
# 8. Feature engineering - outlier handling, encoding, scaling
# 9. Feature selection - SelectKBest with f_regression
# 10. Train/test split and save
# ------------------------------------------------------------

# ============================================================
# Step 1: Load the Dataset
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(
	"https://breathecode.herokuapp.com/asset/internal-link?id=418&path=demographic_health_data.csv"
)
print(f"Shape: {df.shape}")
print(df.head())

# ============================================================
# Step 2: High Level View
# ============================================================

# Data types and non-null counts
df.info(show_counts=True)

# Preview all column names grouped by type
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

print(f"Numerical columns: ({len(num_cols)}):")
for entry in num_cols:
	print(f"  {entry}")

print(f"\nCategorical columns: ({len(cat_cols)}):")
for entry in cat_cols:
	print(f"  {entry}")

# ============================================================
# Step 3: Data Cleaning
# ============================================================

# 3a. Drop columns that should not be used as features.
#
# This dataset has several categories of columns we should drop
# before modeling:
# - ID / geography columns
# - Heart disease leakage columns
# - Other disease CI bounds
# - Raw count columns for other conditions
# - Other prevalence outcome columns
# - Duplicate info columns where percentages/rates are retained
# - Median income CI bounds

# Columns to drop
drop_cols = [
	# Identifiers
	"fips", "STATE_FIPS",

	# Heart disease leakage (derived from target)
	"Heart disease_prevalence",
	"Heart disease_Lower 95% CI",
	"Heart disease_Upper 95% CI",

	# CI bounds for all other conditions
	"anycondition_Lower 95% CI", "anycondition_Upper 95% CI",
	"Obesity_Lower 95% CI", "Obesity_Upper 95% CI",
	"COPD_Lower 95% CI", "COPD_Upper 95% CI",
	"diabetes_Lower 95% CI", "diabetes_Upper 95% CI",
	"CKD_Lower 95% CI", "CKD_Upper 95% CI",

	# Raw count columns for other conditions (redundant with prevalence)
	"anycondition_number", "Obesity_number",
	"COPD_number", "diabetes_number", "CKD_number",

	# Peer disease prevalence columns (outcomes, not predictors)
	"anycondition_prevalence", "diabetes_prevalence",
	"COPD_prevalence", "CKD_prevalence",

	# Raw age group population counts (% equivalents retained)
	"0-9", "10-19", "20-29", "30-39", "40-49",
	"50-59", "60-69", "70-79", "80+", "Population Aged 60+",

	# Raw race population counts (% equivalents retained)
	"White-alone pop", "Black-alone pop",
	"Native American/American Indian-alone pop",
	"Asian-alone pop", "Hawaiian/Pacific Islander-alone pop",
	"Two or more races pop",

	# Median income CI bounds
	"CI90LBINC_2018", "CI90UBINC_2018",

	# Raw employment counts (rate retained)
	"Civilian_labor_force_2018", "Employed_2018", "Unemployed_2018",

	# Raw poverty count (% retained)
	"POVALL_2018",

	# Raw education counts (% equivalents retained)
	"Less than a high school diploma 2014-18",
	"High school diploma only 2014-18",
	"Some college or associate's degree 2014-18",
	"Bachelor's degree or higher 2014-18",

	# Population aged 18+ (correlated with TOT_POP, which we keep)
	"county_pop2018_18 and older",
]

# Only drop columns that actually exist
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

print(f"Columns remaining: {df.shape[1]}")
print(list(df.columns))

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)

# Check for nulls
null_counts = df.isnull().sum()
print(f"\nColumns with nulls:")
print(null_counts[null_counts > 0].sort_values(ascending=False))

# Drop rows where target is null
target = "Heart disease_number"
df = df.dropna(subset=[target]).reset_index(drop=True)

# Fill remaining nulls with column median
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print(f"Nulls remaining: {df.isnull().sum().sum()}")
print(f"Final shape: {df.shape}")

# ============================================================
# Step 4: Descriptive Statistics
# ============================================================

print(df.describe())

# ============================================================
# Step 5: Univariate Analysis
# ============================================================

# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df[target], bins=40, edgecolor="black")
axes[0].set_title("Heart disease_number")
axes[0].set_xlabel("Number of cases")
axes[0].set_ylabel("County count")

axes[1].boxplot(df[target].dropna())
axes[1].set_title("Heart disease_number")
axes[1].set_ylabel("Number of cases")

plt.suptitle("Target Variable: Heart disease_number")
plt.tight_layout()
plt.show()

# Key sociodemographic features - histograms
key_features = [
	"TOT_POP", "PCTPOVALL_2018", "Obesity_prevalence",
	"Unemployment_rate_2018", "MEDHHINC_2018",
	"Percent of adults with less than a high school diploma 2014-18",
	"Percent of adults with a bachelor's degree or higher 2014-18",
	"% Black-alone", "80+ y/o % of total pop",
	"Active Physicians per 100000 Population 2018 (AAMC)",
]
key_features = [c for c in key_features if c in df.columns]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(key_features):
	axes[i].hist(df[col].dropna(), bins=30)
	axes[i].set_title(col[:35], fontsize=8)

for j in range(i + 1, len(axes)):
	fig.delaxes(axes[j])

plt.suptitle("Key Feature Distributions", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# ============================================================
# Step 6: Univariate Analysis - Findings
#
# - Heart disease_number (target): Heavily right-skewed. Most
#   counties have relatively few cases, but a long tail of large,
#   populous counties has very high counts. TOT_POP is critical.
# - TOT_POP: Also right-skewed and strongly related to target.
# - Obesity_prevalence: Right-skewed with strong variation.
# - PCTPOVALL_2018: Right-skewed with a high-poverty tail.
# - MEDHHINC_2018: Right-skewed income distribution.
# - % Black-alone: Heavily right-skewed.
# - 80+ y/o % of total pop: Older age share is strongly predictive.
# - Unemployment_rate_2018: Mostly 3-8%, with outliers above 15%.
# - Physician supply: Right-skewed; many rural counties are underserved.
# ============================================================

# ============================================================
# Step 7: Multivariate Analysis
# ============================================================

# Correlation heatmap
heatmap_features = [
	"Heart disease_number", "TOT_POP", "Obesity_prevalence",
	"PCTPOVALL_2018", "MEDHHINC_2018", "Unemployment_rate_2018",
	"Percent of adults with less than a high school diploma 2014-18",
	"Percent of adults with a bachelor's degree or higher 2014-18",
	"% Black-alone", "80+ y/o % of total pop",
	"Active Physicians per 100000 Population 2018 (AAMC)",
	"Urban_rural_code", "R_death_2018"
]
heatmap_features = [c for c in heatmap_features if c in df.columns]

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(
	df[heatmap_features].corr(),
	annot=True, fmt=".2f", ax=ax,
	cmap="coolwarm", linewidths=0.5
)
plt.title("Correlation Heatmap - Key Features vs Target", fontsize=13)
plt.tight_layout()
plt.show()

# Scatter plots - top predictors vs Heart disease_number
scatter_features = [
	"TOT_POP", "Obesity_prevalence", "PCTPOVALL_2018",
	"MEDHHINC_2018", "% Black-alone", "80+ y/o % of total pop",
]
scatter_features = [c for c in scatter_features if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(scatter_features):
	axes[i].scatter(df[col], df[target], alpha=0.2, edgecolors="none", s=10)
	axes[i].set_xlabel(col[:40], fontsize=8)
	axes[i].set_ylabel("Heart disease_number")
	axes[i].set_title(f"{col[:30]} vs target", fontsize=8)

plt.suptitle("Scatter Plots - Key Features vs Heart Disease Number", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# ============================================================
# Step 8: Multivariate Analysis
#
# Strongest correlations with Heart disease_number:
# - TOT_POP
# - Obesity_prevalence
# - % Black-alone
# - PCTPOVALL_2018
# - 80+ y/o % of total pop
# - R_death_2018
# - MEDHHINC_2018 (inverse)
# - Percent of adults with a bachelor's degree or higher (inverse)
# - Active Physicians per 100000 Population (inverse)
# ============================================================

# ============================================================
# Step 9: Feature Engineering
# ============================================================

# Outlier handling (IQR method)
features_to_cap = ["TOT_POP", "MEDHHINC_2018", "GQ_ESTIMATES_2018"]
features_to_cap = [c for c in features_to_cap if c in df.columns]

for col in features_to_cap:
	stats = df[col].describe()
	iqr = stats["75%"] - stats["25%"]
	upper = stats["75%"] + 1.5 * iqr
	before = len(df)
	df = df[df[col] <= upper].reset_index(drop=True)
	print(f"{col}: capped at {upper:.1f}, removed {before - len(df)} rows")

print(f"\nRows remaining: {len(df)}")

# Drop any remaining string columns (e.g. county name)
string_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Dropping string columns: {string_cols}")
df = df.drop(columns=string_cols)

# Feature scaling: MinMaxScaler on all feature columns
from sklearn.preprocessing import MinMaxScaler

feature_cols = [c for c in df.columns if c != target]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
	scaler.fit_transform(df[feature_cols]),
	columns=feature_cols,
	index=df.index
)

# Add target back unscaled
df_scaled[target] = df[target].values

print(f"Shape after scaling: {df_scaled.shape}")
print(df_scaled.head())

# ============================================================
# Step 10: Feature Selection
#
# Note on feature selection with Lasso:
# Lasso applies L1 regularization and can zero weak feature
# coefficients, so it already performs feature selection during
# training. We still run SelectKBest to remove clear noise upfront,
# but keep a generous k=15.
# ============================================================

from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split

X = df_scaled.drop(target, axis=1)
y = df_scaled[target]

# Train/test split BEFORE SelectKBest to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# k=15 is generous since Lasso will further prune via regularization
selection_model = SelectKBest(f_regression, k=15)
selection_model.fit(X_train, y_train)

ix = selection_model.get_support()
selected_features = X_train.columns.values[ix]
print(f"Selected features ({len(selected_features)}):")
for f in selected_features:
	print(f"  {f}")

X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=selected_features)
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=selected_features)

# ============================================================
# Step 11: Save Results
# ============================================================

import os

# Re-attach target column
X_train_sel[target] = list(y_train)
X_test_sel[target] = list(y_test)

# Save to ../data/processed/
base_dir = os.path.dirname(os.path.abspath("__file__"))
processed_dir = os.path.join(base_dir, "../data/processed")
os.makedirs(processed_dir, exist_ok=True)

X_train_sel.to_csv(os.path.join(processed_dir, "clean_train_data.csv"), index=False)
X_test_sel.to_csv(os.path.join(processed_dir, "clean_test_data.csv"), index=False)

print("Files saved successfully.")

print(X_train_sel.head())

# ============================================================
# Logistic Regression Model
# ============================================================

train_data = pd.read_csv("../data/processed/clean_train_data.csv")
test_data = pd.read_csv("../data/processed/clean_test_data.csv")

print(train_data.head())

X_train = train_data.drop(["Heart disease_number"], axis=1)
y_train = train_data["Heart disease_number"]
X_test = test_data.drop(["Heart disease_number"], axis=1)
y_test = test_data["Heart disease_number"]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"Intercep (a): {model.intercept_}")
print(f"Coefficients: {model.coef_}")

print("\nThis is our prediction:")
y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error, r2_score

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

# ============================================================
# Model optimization
# ============================================================

from sklearn.linear_model import Lasso

alpha = 1.0
lasso_model = Lasso(alpha=alpha)

# Training the model
lasso_model.fit(X_train, y_train)

# Evaluate performance on the test data
score = lasso_model.score(X_test, y_test)
print("Coefficients:", lasso_model.coef_)
print("R2 score:", score)

from pickle import dump

dump(lasso_model, open("../models/lasso_alpha-1.0.sav", "wb"))
