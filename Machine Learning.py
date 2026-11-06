import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "Online Retail.xlsx"         
df = pd.read_excel(file_path)
print(df.head())
print("Shape:", df.shape)
print("\nColumns:", df.columns)
print("\n Dataset information")
df.info()
print("\nSatistical Summary:", df.describe())

# Data Quality Analysis
print("\nMissing values:", df.isnull().sum())
print("\nDuplicate Records:")
print(df.duplicated().sum())
print("\nNegative Quantities:")
print((df['Quantity'] < 0).sum())
print("\nNegative Prices:")
print((df['UnitPrice'] < 0).sum())
print("\nZero Prices:")
print((df['UnitPrice'] == 0).sum())
print("\nCancelled Transactions:")
print(df['InvoiceNo'].astype(str).str.startswith('C').sum())
print("\nUnique Products:")
print(df['StockCode'].nunique())

# Data Cleaning Code
clean_df = df.copy()
clean_df = clean_df.drop_duplicates()
clean_df = clean_df[~clean_df['InvoiceNo'].astype(str).str.startswith('C')]
clean_df = clean_df[clean_df['Quantity'] > 0]
clean_df = clean_df[clean_df['UnitPrice'] > 0]
clean_df['Description'] = clean_df['Description'].fillna('Unknown')
clean_df['TotalSales'] = (clean_df['Quantity'] * clean_df['UnitPrice'])
print("Original Dataset Shape:", df.shape)
print("Cleaned Dataset Shape:", clean_df.shape)
print("\nMissing Values After Cleaning:")
print(clean_df.isnull().sum())
print("\nDuplicate Records After Cleaning:")
print(clean_df.duplicated().sum())
print("\nCleaned Dataset Preview:")
print(clean_df.head())
clean_df.to_excel("data/cleaned_online retail.xlsx", index= False)
print("Cleaned dataset saved successfully!")

# Feature Engineering
model_df = clean_df.copy()
model_df['InvoiceDate'] = pd.to_datetime(model_df['InvoiceDate'])
model_df['StockCode'] = model_df['StockCode'].astype(str)
model_df['Date'] = model_df['InvoiceDate'].dt.normalize()
top_products = (model_df.groupby('StockCode')['Quantity'].sum().nlargest(20).index)
model_df = model_df[model_df['StockCode'].isin(top_products)]
daily = model_df.groupby(
    ['StockCode', 'Date']
)['Quantity'].sum().reset_index()
all_dates = pd.date_range( daily['Date'].min(),daily['Date'].max(),freq='D')
all_products = daily['StockCode'].unique()
full_index = pd.MultiIndex.from_product(
    [all_products, all_dates],
    names=['StockCode', 'Date']
)
daily = (
    daily.set_index(['StockCode', 'Date'])
    .reindex(full_index, fill_value=0)
    .reset_index()
)
daily.rename(
    columns={'Quantity': 'DailyDemand'},
    inplace=True
)
daily['Day'] = daily['Date'].dt.day
daily['Month'] = daily['Date'].dt.month
daily['DayOfWeek'] = daily['Date'].dt.dayofweek
daily['IsWeekend'] = (
    daily['DayOfWeek'] >= 5
).astype(int)
daily['Lag_1'] = (
    daily.groupby('StockCode')['DailyDemand'].shift(1)
)

daily['Lag_7'] = (
    daily.groupby('StockCode')['DailyDemand'].shift(7)
)
daily['Rolling_Mean_7'] = (
    daily.groupby('StockCode')['DailyDemand']
    .transform(lambda x: x.shift(1).rolling(7).mean())
)
daily.dropna(inplace=True)
daily.reset_index(drop=True, inplace=True)
print("\nFeature Engineered Dataset:")
print(daily.head())

print("\nDataset Shape:")
print(daily.shape)

print("\nFinal Columns:")
print(daily.columns)

daily.to_excel(
    'data/retail_demand_features.xlsx',
    index=False
)
print("\nFeature Engineering Completed!")

# Exploratory Data Analysis
sns.set_theme(style="whitegrid")
product_demand = (
    daily.groupby('StockCode')['DailyDemand']
    .sum()
    .sort_values(ascending=False)
)
plt.figure(figsize=(12, 6))
product_demand.plot(kind='bar')
plt.title("Total Demand by Product")
plt.xlabel("Product Code")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
daily_trend = daily.groupby('Date')['DailyDemand'].sum()
plt.figure(figsize=(14, 6))
plt.plot(daily_trend.index, daily_trend.values)
plt.title("Daily Product Demand Trend")
plt.xlabel("Date")
plt.ylabel("Total Daily Demand")
plt.tight_layout()
plt.show()
weekday_demand = daily.groupby(
    'DayOfWeek'
)['DailyDemand'].mean()
plt.figure(figsize=(10, 5))
weekday_demand.plot(kind='bar')
plt.title("Average Demand by Day of Week")
plt.xlabel("Day (0 = Monday, 6 = Sunday)")
plt.ylabel("Average Demand")
plt.tight_layout()
plt.show()
daily['YearMonth'] = daily['Date'].dt.to_period('M').astype(str)
monthly_demand = daily.groupby(
    'YearMonth'
)['DailyDemand'].sum()
plt.figure(figsize=(14, 6))
monthly_demand.plot(kind='line', marker='o')
plt.title("Monthly Product Demand")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("EDA Completed Successfully!")

# Preparing Data for Machine Learning
from sklearn.preprocessing import LabelEncoder
ml_df = daily.copy()
encoder = LabelEncoder()
ml_df['StockCode'] = encoder.fit_transform( ml_df['StockCode'].astype(str))
X = ml_df[['StockCode','Day','Month','DayOfWeek','IsWeekend','Lag_1','Lag_7','Rolling_Mean_7']]

# Target variable
y = ml_df['DailyDemand']

# CORRECT CHRONOLOGICAL SPLIT

# Sort data by date
ml_df = ml_df.sort_values(
    by='Date'
).reset_index(drop=True)

# Recreate features after sorting
X = ml_df[
    ['StockCode', 'Day', 'Month',
     'DayOfWeek', 'IsWeekend',
     'Lag_1', 'Lag_7', 'Rolling_Mean_7']
]
y = ml_df['DailyDemand']
# Split using unique dates
dates = sorted(ml_df['Date'].unique())
cutoff_index = int(len(dates) * 0.80)
cutoff_date = dates[cutoff_index]
train_mask = ml_df['Date'] < cutoff_date
test_mask = ml_df['Date'] >= cutoff_date
X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]
print("Chronological split completed!")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)
print("Training ends:", ml_df.loc[train_mask, 'Date'].max())
print("Testing starts:", ml_df.loc[test_mask, 'Date'].min())

# Model Training
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

print("Libraries imported successfully!")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("All models trained successfully!")
print("Model Training Results")
print("----------------------")
print("Linear Regression: Trained")
print("Decision Tree: Trained")
print("Random Forest: Trained")

# MODEL PREDICTIONS
lr_predictions = lr_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
print("Predictions generated successfully!")
print("\nFirst 10 Predictions:")
print("\nActual Demand:")
print(y_test.head(10).values)
print("\nLinear Regression:")
print(lr_predictions[:10])
print("\nDecision Tree:")
print(dt_predictions[:10])
print("\nRandom Forest:")
print(rf_predictions[:10])

# MODEL EVALUATION
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np
import pandas as pd

# Evaluate Linear Regression
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_predictions)

# Evaluate Decision Tree
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_predictions)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

# Create comparison table
results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree',
        'Random Forest'
    ],
    'MAE': [lr_mae, dt_mae, rf_mae],
    'MSE': [lr_mse, dt_mse, rf_mse],
    'RMSE': [lr_rmse, dt_rmse, rf_rmse],
    'R2 Score': [lr_r2, dt_r2, rf_r2]
})
print("\nMODEL EVALUATION RESULTS")
print(results.round(3))

# MODEL DIAGNOSTICS
print("\nTRAINING TARGET SUMMARY")
print(y_train.describe())

print("\nTESTING TARGET SUMMARY")
print(y_test.describe())

print("\nFEATURE SUMMARY")
print(X_train.describe())

print("\nLARGEST ACTUAL DEMAND VALUES")
print(y_test.sort_values(ascending=False).head(10))

print("\nLARGEST RANDOM FOREST PREDICTIONS")
print(
    pd.Series(rf_predictions)
    .sort_values(ascending=False)
    .head(10)
)

# INVESTIGATE EXTREME DEMAND
print("\nTOP 15 TRAINING DEMAND RECORDS")
top_demand = ml_df.loc[
    y_train.nlargest(15).index,
    ['StockCode', 'Date', 'DailyDemand']
]
print(top_demand.to_string(index=False))
print("\nNUMBER OF EXTREME VALUES")
print("Demand above 1000:",(y_train > 1000).sum())
print("Demand above 5000:",(y_train > 5000).sum())
print("Demand above 10000:",(y_train > 10000).sum())
print("\nTARGET SKEWNESS")
print("Training:", y_train.skew())
print("Testing:", y_test.skew())

# CHECK CHRONOLOGICAL SPLIT
train_dates = ml_df.loc[X_train.index, 'Date']
test_dates = ml_df.loc[X_test.index, 'Date']

print("\nTRAINING DATE RANGE")
print("Start:", train_dates.min())
print("End:", train_dates.max())

print("\nTESTING DATE RANGE")
print("Start:", test_dates.min())
print("End:", test_dates.max())

print("\nCHRONOLOGICAL SPLIT CHECK")

if train_dates.max() < test_dates.min():
    print("Correct chronological split!")
else:
    print("Warning: Data is not properly separated by date!")

# INVESTIGATE EXTREME DEMAND VALUES
print("\nDEMAND DISTRIBUTION ANALYSIS")

print("\nTraining Demand Quantiles:")
print(y_train.quantile([0.50, 0.75, 0.90, 0.95, 0.99, 1.00]))

print("\nExtreme Demand Counts:")

print("Above 500:", (y_train > 500).sum())
print("Above 1000:", (y_train > 1000).sum())
print("Above 5000:", (y_train > 5000).sum())
print("Above 10000:", (y_train > 10000).sum())
print("\nTop 10 Highest Training Demand Values:")
print(y_train.nlargest(10))

# LOG TRANSFORMATION
import numpy as np

# Transform training and testing targets
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
print("\nLOG TRANSFORMATION COMPLETED!")
print("\nOriginal Training Demand:")
print(y_train.describe())
print("\nTransformed Training Demand:")
print(y_train_log.describe())
print("\nOriginal Maximum:", y_train.max())
print("Transformed Maximum:", y_train_log.max())

# TRAIN MODELS WITH LOG TRANSFORMATION
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
print("\nTRAINING IMPROVED MODELS...")

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_log)

# Model 2: Decision Tree
dt_model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
dt_model.fit(X_train, y_train_log)

# Model 3: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train_log)
print("All improved models trained successfully!")

# GENERATE PREDICTIONS
print("\nGENERATING PREDICTIONS...")

# Predictions on logarithmic scale
lr_log_predictions = lr_model.predict(X_test)
dt_log_predictions = dt_model.predict(X_test)
rf_log_predictions = rf_model.predict(X_test)

# Convert predictions back to original scale
lr_predictions = np.expm1(lr_log_predictions)
dt_predictions = np.expm1(dt_log_predictions)
rf_predictions = np.expm1(rf_log_predictions)

# Demand cannot be negative
lr_predictions = np.maximum(0, lr_predictions)
dt_predictions = np.maximum(0, dt_predictions)
rf_predictions = np.maximum(0, rf_predictions)
print("Predictions generated successfully!")

# MODEL EVALUATION
print("\nEVALUATING IMPROVED MODELS...")
models_predictions = {
    "Linear Regression": lr_predictions,
    "Decision Tree": dt_predictions,
    "Random Forest": rf_predictions
}
results = []
for model_name, predictions in models_predictions.items():
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })
evaluation_df = pd.DataFrame(results)
print("\nIMPROVED MODEL EVALUATION RESULTS")
print(evaluation_df.round(3))

# SELECT BEST MODEL
best_model_name = evaluation_df.loc[
   evaluation_df["MAE"].idxmin(), "Model"
]
trained_models = {
    "Linear Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}
best_model = trained_models[best_model_name]
print("\nBEST MODEL SELECTED:", best_model_name)
joblib.dump(best_model, "best_demand_model.pkl")
print("\nBest model saved successfully!")
model_information = {
    "model_name": best_model_name,
    "features": list(X_train.columns),
    "target_transformation": "log1p",
    "model_type": "Demand Prediction"
}
joblib.dump(model_information, "model_information.pkl")
print("Model information saved successfully!")
evaluation_df.to_csv(
    "improved_model_evaluation.csv",
    index=False
)
print("Evaluation results saved successfully!")
prediction_results = pd.DataFrame({
    "Actual_Demand": y_test.values,
    "Linear_Regression": lr_predictions,
    "Decision_Tree": dt_predictions,
    "Random_Forest": rf_predictions
})
prediction_results.to_csv(
    "demand_predictions.csv",
    index=False
)
print("Prediction results saved successfully!")
print("\n================================")
print("MACHINE LEARNING PIPELINE COMPLETED")
print("Models trained: 3")
print("Predictions generated: Yes")
print("Model evaluation: Completed")
print("Best model:", best_model_name)
print("Model saved: Yes")
print("\nALL ML TASKS COMPLETED SUCCESSFULLY!")