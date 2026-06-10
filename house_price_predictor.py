# HOUSE PRICE PREDICTOR
# Dataset: Kaggle House Prices (train.csv)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

df = pd.read_csv('train.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.shape[1]}")
print(f"\nFirst look at target (SalePrice):")
print(df['SalePrice'].describe())

# EDA 

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('House Price EDA — Key Insights', fontsize=16, fontweight='bold')

# Plot 1: SalePrice distribution
axes[0, 0].hist(df['SalePrice'], bins=50, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Sale Price Distribution')
axes[0, 0].set_xlabel('Sale Price ($)')
axes[0, 0].set_ylabel('Count')

# Plot 2: Log-transformed SalePrice (more normal — better for regression)
axes[0, 1].hist(np.log1p(df['SalePrice']), bins=50, color='coral', edgecolor='white')
axes[0, 1].set_title('Log(Sale Price) — More Normal')
axes[0, 1].set_xlabel('log(Sale Price)')
axes[0, 1].set_ylabel('Count')

# Plot 3: GrLivArea vs SalePrice (scatter)
axes[0, 2].scatter(df['GrLivArea'], df['SalePrice'], alpha=0.4, color='green', s=10)
axes[0, 2].set_title('Living Area vs Sale Price')
axes[0, 2].set_xlabel('Above Ground Living Area (sqft)')
axes[0, 2].set_ylabel('Sale Price ($)')

# Plot 4: Missing values (top 20 columns)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False).head(20)
axes[1, 0].barh(missing.index, missing.values, color='salmon')
axes[1, 0].set_title('Top 20 Columns with Missing Values')
axes[1, 0].set_xlabel('Count of Missing')

# Plot 5: Overall Quality vs SalePrice (box plot)
df.boxplot(column='SalePrice', by='OverallQual', ax=axes[1, 1])
axes[1, 1].set_title('Overall Quality vs Sale Price')
axes[1, 1].set_xlabel('Overall Quality (1-10)')
axes[1, 1].set_ylabel('Sale Price ($)')
plt.sca(axes[1, 1])
plt.title('Overall Quality vs Sale Price')

# Plot 6: Correlation heatmap (top 10 numeric features)
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()['SalePrice'].abs().sort_values(ascending=False)[1:11]
top_feats = corr.index.tolist() + ['SalePrice']
corr_matrix = df[top_feats].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 2], cbar=True, square=False, annot_kws={'size': 8})
axes[1, 2].set_title('Correlation Heatmap (Top Features)')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 features correlated with SalePrice:")
print(corr.round(3).to_string())

# FEATURE ENGINEERING

df_fe = df.copy()

# -- 3a. Drop ID column
df_fe.drop('Id', axis=1, inplace=True)

# -- 3b. Remove outliers (GrLivArea > 4000 sqft with low price — known Kaggle issue)
df_fe = df_fe[~((df_fe['GrLivArea'] > 4000) & (df_fe['SalePrice'] < 300000))]
print(f"After outlier removal: {df_fe.shape[0]} rows")

# -- 3c. Log-transform target (makes it more normal, improves regression)
df_fe['SalePrice'] = np.log1p(df_fe['SalePrice'])
print("✅ Log-transformed SalePrice")

# -- 3d. Create new useful features
df_fe['TotalSF'] = df_fe['TotalBsmtSF'] + df_fe['1stFlrSF'] + df_fe['2ndFlrSF']
df_fe['TotalBath'] = (df_fe['FullBath'] + 0.5 * df_fe['HalfBath'] +
                      df_fe['BsmtFullBath'] + 0.5 * df_fe['BsmtHalfBath'])
df_fe['HouseAge'] = df_fe['YrSold'] - df_fe['YearBuilt']
df_fe['RemodAge'] = df_fe['YrSold'] - df_fe['YearRemodAdd']
df_fe['HasGarage'] = (df_fe['GarageArea'] > 0).astype(int)
df_fe['HasPool'] = (df_fe['PoolArea'] > 0).astype(int)
df_fe['HasFireplace'] = (df_fe['Fireplaces'] > 0).astype(int)
print("✅ Created 7 new features: TotalSF, TotalBath, HouseAge, RemodAge, HasGarage, HasPool, HasFireplace")

# -- 3e. Handle missing values
# Numeric: fill with median
numeric_cols = df_fe.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_fe[col].isnull().sum() > 0:
        df_fe[col].fillna(df_fe[col].median(), inplace=True)

# Categorical: fill with 'None' (means feature is absent, e.g. no garage)
cat_cols = df_fe.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df_fe[col].isnull().sum() > 0:
        df_fe[col].fillna('None', inplace=True)

print(f"✅ Missing values after fill: {df_fe.isnull().sum().sum()}")

# -- 3f. Encode categorical columns (Label Encoding for tree models, fine for Ridge too)
le = LabelEncoder()
for col in cat_cols:
    df_fe[col] = le.fit_transform(df_fe[col].astype(str))
print(f"✅ Encoded {len(cat_cols)} categorical columns")

print(f"\nFinal feature count: {df_fe.shape[1] - 1} features")

# ============================================================
# STEP 4 — SELECT TOP FEATURES (keep it clean for Streamlit UI)
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: Feature Selection")
print("=" * 55)

# Top features to use — mix of engineered + original
TOP_FEATURES = [
    'TotalSF', 'OverallQual', 'GrLivArea', 'GarageCars',
    'TotalBath', 'YearBuilt', 'TotalBsmtSF', 'HouseAge',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd',
    'GarageArea', 'LotArea', 'OverallCond', 'MSSubClass',
    'BedroomAbvGr', 'KitchenAbvGr', 'HasFireplace', 'HasGarage'
]

X = df_fe[TOP_FEATURES]
y = df_fe['SalePrice']

print(f"Using {len(TOP_FEATURES)} features")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ============================================================
# STEP 5 — TRAIN/TEST SPLIT + SCALE
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: Train/Test Split & Scaling")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ============================================================
# STEP 6 — TRAIN LINEAR REGRESSION vs RIDGE (compare both)
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Model Training — Linear vs Ridge")
print("=" * 55)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)

# Ridge Regression (alpha=10 is a good default, tune if needed)
ridge = Ridge(alpha=10)
ridge.fit(X_train_scaled, y_train)
ridge_preds = ridge.predict(X_test_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
ridge_r2 = r2_score(y_test, ridge_preds)

# Cross-validation scores
lr_cv = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='r2').mean()
ridge_cv = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2').mean()

print(f"\n{'Model':<20} {'Test RMSE':<15} {'Test R²':<12} {'CV R² (5-fold)'}")
print("-" * 62)
print(f"{'Linear Regression':<20} {lr_rmse:<15.4f} {lr_r2:<12.4f} {lr_cv:.4f}")
print(f"{'Ridge (alpha=10)':<20} {ridge_rmse:<15.4f} {ridge_r2:<12.4f} {ridge_cv:.4f}")
print(f"\n→ Ridge wins on CV score. Using Ridge for deployment.")

# Convert RMSE back to actual dollar value for intuition
rmse_dollars = np.expm1(ridge_rmse + np.mean(y_test)) - np.expm1(np.mean(y_test))
print(f"→ Ridge R²: {ridge_r2:.4f}  (model explains {ridge_r2*100:.1f}% of price variance)")

# ============================================================
# STEP 7 — FEATURE IMPORTANCE PLOT
# ============================================================
print("\n" + "=" * 55)
print("STEP 7: Feature Importance")
print("=" * 55)

coef_df = pd.DataFrame({
    'Feature': TOP_FEATURES,
    'Coefficient': np.abs(ridge.coef_)
}).sort_values('Coefficient', ascending=True)

plt.figure(figsize=(10, 8))
colors = ['#2196F3' if c > coef_df['Coefficient'].median() else '#90CAF9'
          for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
plt.title('Ridge Regression — Feature Importance (|Coefficient|)', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance plot saved as feature_importance.png")

# ============================================================
# STEP 8 — ACTUAL vs PREDICTED PLOT
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, ridge_preds, alpha=0.5, color='steelblue', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual log(SalePrice)')
plt.ylabel('Predicted log(SalePrice)')
plt.title(f'Actual vs Predicted (Ridge R² = {ridge_r2:.3f})', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Actual vs Predicted plot saved as actual_vs_predicted.png")

# ============================================================
# STEP 9 — SAVE MODEL + SCALER + FEATURE LIST
# ============================================================
print("\n" + "=" * 55)
print("STEP 9: Saving Model")
print("=" * 55)

joblib.dump(ridge, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(TOP_FEATURES, 'features.pkl')

print("✅ Saved: model.pkl")
print("✅ Saved: scaler.pkl")
print("✅ Saved: features.pkl")

# ============================================================
# STEP 10 — QUICK SANITY CHECK (predict one sample)
# ============================================================
print("\n" + "=" * 55)
print("STEP 10: Sanity Check")
print("=" * 55)

sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)
pred_log = ridge.predict(sample_scaled)[0]
pred_price = np.expm1(pred_log)
actual_price = np.expm1(y_test.iloc[0])

print(f"Sample prediction:  ${pred_price:,.0f}")
print(f"Actual price:       ${actual_price:,.0f}")
print(f"Difference:         ${abs(pred_price - actual_price):,.0f}")

print("\n" + "=" * 55)
print("✅ DAY 1 COMPLETE!")
print("=" * 55)
print("""
Files created:
  model.pkl            ← Ridge regression model
  scaler.pkl           ← StandardScaler (needed for app)
  features.pkl         ← List of 20 feature names
  eda_plots.png        ← 6-panel EDA visualization
  feature_importance.png
  actual_vs_predicted.png

Next → Day 2: Build Streamlit app using model.pkl
""")
