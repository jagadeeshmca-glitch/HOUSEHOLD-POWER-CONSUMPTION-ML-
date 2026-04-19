# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    print("⚠ XGBoost not installed")
    xgb_available = False


# =========================
# STEP 1: LOAD DATASET
# =========================
df = pd.read_csv('household_power_consumption.csv')

print("\n====================================")
print("✅ DATASET LOADED SUCCESSFULLY")
print("====================================")


# =========================
# DATA PREVIEW
# =========================
print("\n📌 DATA PREVIEW")
print(df.head())

print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nMissing Values (Before):")
print(df.isnull().sum())


# =========================
# CLEANING
# =========================
df.columns = df.columns.str.strip()
df.drop(['Date', 'Time', 'index'], axis=1, errors='ignore', inplace=True)

df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')


# =========================
# MEMORY OPTIMIZATION
# =========================
for col in df.select_dtypes(include=['int64']):
    df[col] = df[col].astype('int32')

for col in df.select_dtypes(include=['float64']):
    df[col] = df[col].astype('float32')

print("\n✅ Memory Optimized")


# =========================
# HANDLE MISSING VALUES
# =========================
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing Values (After):")
print(df.isnull().sum())


# =========================
# TARGET SPLIT
# =========================
target = 'Global_active_power'

X = df.drop(target, axis=1)
y = df[target]

print("\nOriginal Features:", X.shape[1])


# =========================
# FEATURE SELECTION
# =========================
var_selector = VarianceThreshold(threshold=0)
X = var_selector.fit_transform(X)

selector = SelectKBest(score_func=f_regression, k=6)
X = selector.fit_transform(X, y)

print("Selected Features:", X.shape[1])


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))


# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# FUNCTION: GRAPH
# =========================
def plot_model_graph(y_test, y_pred, model_name):
    plt.figure()
    plt.plot(y_test.values[:100], label="Actual")
    plt.plot(y_pred[:100], label="Predicted")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.xlabel("Samples")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.savefig(f"{model_name}.png")


# =========================
# MODEL 1: LINEAR REGRESSION
# =========================
print("\n====================================")
print("LINEAR REGRESSION MODEL")
print("====================================")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
lr_r2 = r2_score(y_test, pred_lr)
lr_mae = mean_absolute_error(y_test, pred_lr)
print(f"Accuracy (R2): {lr_r2:.4f}")
print(f"MAE          : {lr_mae:.4f}")
plot_model_graph(y_test, pred_lr, "Linear_Regression")

# =========================
# MODEL 2: DECISION TREE
# =========================
print("\n====================================")
print("DECISION TREE MODEL")
print("====================================")
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
dt_r2 = r2_score(y_test, pred_dt)
dt_mae = mean_absolute_error(y_test, pred_dt)
print(f"Accuracy (R2): {dt_r2:.4f}")
print(f"MAE          : {dt_mae:.4f}")
plot_model_graph(y_test, pred_dt, "Decision_Tree")


# =========================
# MODEL 3: RANDOM FOREST
# =========================
print("\n====================================")
print("RANDOM FOREST MODEL")
print("====================================")
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
rf_r2 = r2_score(y_test, pred_rf)
rf_mae = mean_absolute_error(y_test, pred_rf)
print(f"Accuracy (R2): {rf_r2:.4f}")
print(f"MAE          : {rf_mae:.4f}")
plot_model_graph(y_test, pred_rf, "Random_Forest")


# =========================
# MODEL 5: XGBOOST
# =========================
if xgb_available:
    print("\n====================================")
    print("XGBOOST MODEL")
    print("====================================")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    xgb_r2 = r2_score(y_test, pred_xgb)
    xgb_mae = mean_absolute_error(y_test, pred_xgb)
    print(f"Accuracy (R2): {xgb_r2:.4f}")
    print(f"MAE          : {xgb_mae:.4f}")
    plot_model_graph(y_test, pred_xgb, "XGBoost")


# =========================
# SAMPLE PREDICTION
# =========================
print("\n====================================")
print("SAMPLE PREDICTION")
print("====================================")

sample = X_test[0].reshape(1, -1)
prediction = rf.predict(sample)

print(f"Predicted Energy Consumption: {prediction[0]:.4f} kWh")

# =========================
# MODEL COMPARISON BAR CHART
# =========================

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

# Store values (use only available models)
models = ["LR", "DT", "RF"]
r2_scores = [lr_r2, dt_r2, rf_r2]

# Add XGBoost if available
if xgb_available:
    models.append("XGB")
    r2_scores.append(xgb_r2)

# Print values
for m, r in zip(models, r2_scores):
    print(f"{m} -> R2 Score: {r:.4f}")

# Create bar chart
plt.figure()

bars = plt.bar(models, r2_scores)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.2f}", ha='center', va='bottom')

plt.title("Model Comparison (R2 Score)")
plt.xlabel("Algorithms")
plt.ylabel("R2 Score")

plt.savefig("model_comparison.png")
# =========================
# MAE COMPARISON CHART
# =========================

mae_scores = [lr_mae, dt_mae, rf_mae]

# Add XGBoost if available
if xgb_available:
    mae_scores.append(xgb_mae)

plt.figure()

bars = plt.bar(models, mae_scores)

# Add values on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.2f}", ha='center', va='bottom')

plt.title("Model Comparison (MAE)")
plt.xlabel("Algorithms")
plt.ylabel("MAE (Error)")

plt.savefig("mae_comparison.png")
# =========================
# R2 COMPARISON (4 MODELS)
# =========================

print("\n====================================")
print("R2 SCORE COMPARISON (4 MODELS)")
print("====================================")

# Fixed 4 models
models = ["LR", "DT", "RF", "XGB"]
r2_scores = [lr_r2, dt_r2, rf_r2, xgb_r2]
# Print values
for m, r in zip(models, r2_scores):
    print(f"{m} -> R2 Score: {r:.4f}")

# Plot graph
plt.figure()

bars = plt.bar(models, r2_scores)

# Add values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.2f}", ha='center', va='bottom')

plt.title("R2 Score Comparison (LR, DT, RF, XGBoost)")
plt.xlabel("Models")
plt.ylabel("R2 Score")

plt.savefig("r2_4models.png")
print("\n====================================")
print("PERFORMANCE ANALYSIS")
print("====================================")

best_r2 = max(r2_scores)
best_model = models[r2_scores.index(best_r2)]

best_mae = min(mae_scores)
best_mae_model = models[mae_scores.index(best_mae)]

print(f"\nBest Model (Highest Accuracy): {best_model} ({best_r2*100:.2f}%)")
print(f"Best Model (Lowest Error)   : {best_mae_model} ({best_mae:.4f})")

print("\nDetailed Comparison:")

for i in range(len(models)):
    print(f"{models[i]} -> Accuracy: {r2_scores[i]*100:.2f}% | MAE: {mae_scores[i]:.4f}")
# Interpretation
print("\nConclusion:")
print(f"{best_model} provides the highest prediction accuracy.")
print(f"{best_mae_model} produces the lowest error.")
print("Ensemble models (RF, XGBoost) generally perform better than basic models.")

# =========================
# SHOW ALL GRAPHS
# =========================
plt.show()
