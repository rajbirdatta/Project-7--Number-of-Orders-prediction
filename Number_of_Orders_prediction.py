import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# I have used the Rossmann Store Sales dataset.

train = pd.read_csv("train.csv")
store = pd.read_csv("store.csv")

df = train.merge(store, on="Store", how="left")

df["Date"] = pd.to_datetime(df["Date"])

df = df[(df["Open"] == 1) & (df["Sales"] > 0)]

df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].max())

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)

df["StoreType"] = df["StoreType"].astype(str).astype("category").cat.codes
df["Assortment"] = df["Assortment"].astype(str).astype("category").cat.codes
df["StateHoliday"] = df["StateHoliday"].astype(str).astype("category").cat.codes

df = df.sort_values("Date")

split_date = "2015-06-01"

train_df = df[df["Date"] < split_date].copy()
test_df = df[df["Date"] >= split_date].copy()

store_avg = train_df.groupby('Store')['Sales'].mean().reset_index()
store_avg.rename(columns={'Sales': 'Store_Avg_Sales'}, inplace=True)

train_df = train_df.merge(store_avg, on='Store', how='left')
test_df = test_df.merge(store_avg, on='Store', how='left')

global_avg = train_df["Store_Avg_Sales"].mean()
test_df["Store_Avg_Sales"] = test_df["Store_Avg_Sales"].fillna(global_avg)

features = [
    "Store", "Promo", "SchoolHoliday",
    "StoreType", "Assortment",
    "CompetitionDistance",
    "Year", "Month", "Day", "WeekOfYear",
    "DayOfWeek", "IsWeekend",
    "Store_Avg_Sales"
]

X_train = train_df[features]
y_train = np.log1p(train_df["Sales"])

X_test = test_df[features]
y_test = test_df["Sales"]

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

pred_log = model.predict(X_test)
predictions = np.expm1(pred_log)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"MAE: {mae:.2f}")
print(f"R2:  {r2:.4f}")
print("-" * 30)

store_id = 1
temp = test_df[test_df["Store"] == store_id]
store_preds = predictions[test_df["Store"] == store_id]

plt.figure(figsize=(12,5))
plt.plot(temp["Date"], temp["Sales"], label="Actual", alpha=0.7)
plt.plot(temp["Date"], store_preds, label="Predicted", linestyle='--')
plt.legend()
plt.xticks(rotation=45)
plt.title(f"Store {store_id} Sales Forecast")
plt.tight_layout()
plt.show()