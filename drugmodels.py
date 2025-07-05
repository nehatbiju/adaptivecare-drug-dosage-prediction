None selected 

Skip to content
Using COIMBATORE INSTITUTE OF TECHNOLOGY Mail with screen readers
praveen 

10 of many
neha proj
Inbox

33 PRAVEEN PV
Jun 17, 2025, 9:54 PM
to me

 label_encoder.pkl
 label_encoder_local.pkl
 labelfinal_encoder55final.pkl
 modeltunedntb.py
 nehaaaa2.py
 xgb_classifier_local.pkl
 xgb_regressor_tuned_local.pkl
 xgbfinal_classifier_modelfinal.pkl
 xgbfinal_regressor_model55final.pkl
 9 Attachments
  •  Scanned by Gmail

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import joblib
import warnings
import gc

warnings.filterwarnings("ignore")

# 📁 Load Data
df = pd.read_csv("cleaned_drugdose_datasetfinal.csv")
df = df.sample(20000, random_state=42)

# 🧹 Remove irrelevant columns
drop_cols = ['DOBMM', 'DOBYY', 'RXBEGMM', 'RXBEGYRX']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# 🧠 Estimate weight fallback
def estimate_weight(age, gender):
    if pd.isna(age) or pd.isna(gender):
        return np.nan
    if isinstance(gender, str):
        gender = gender.lower()
    if gender in ['male', 1, '1', 'm']:
        return 2.5 * age + 15
    elif gender in ['female', 0, '0', 'f']:
        return 2.3 * age + 14
    else:
        return np.nan

df['WEIGHT_KG'] = df.apply(lambda row: estimate_weight(row['AGELAST'], row['SEX.1']), axis=1)

# 🔧 Features + Targets
X = df.drop(columns=['DRUG_LABEL', 'AVG_DAILY_DOSAGE_mg', 'RXNAME', 'RXSTRENG'], errors='ignore')
X = X.select_dtypes(include=np.number)
y_cls = df['DRUG_LABEL']
y_reg = df['AVG_DAILY_DOSAGE_mg']

# 🛠 Filter to top drugs with >55 occurrences
common_drugs = [drug for drug, count in Counter(y_cls).items() if count > 55]
mask = y_cls.isin(common_drugs)
X = X[mask]
y_cls = y_cls[mask]
y_reg = y_reg[mask]

# 🔀 Train-Test Split
X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

# 🏷 Label Encoding
le = LabelEncoder()
y_train_cls_enc = le.fit_transform(y_train_cls)
y_test_cls_enc = le.transform(y_test_cls)
joblib.dump(le, "label_encoder_local.pkl")

# 🔄 Regression Data Split
X_train_reg_core, X_val_reg, y_train_reg_core, y_val_reg = train_test_split(
    X_train, y_train_reg, test_size=0.2, random_state=42
)

gc.collect()

# 🧠 XGBoost Classifier
xgb_cls = XGBClassifier(
    objective='multi:softmax',
    use_label_encoder=False,
    eval_metric='mlogloss',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)
xgb_cls.fit(X_train, y_train_cls_enc)

# 🔍 Hyperparameter Tuning for XGB Regressor
xgb_base = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

param_grid_regressor = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

grid_search_regressor = GridSearchCV(
    xgb_base,
    param_grid_regressor,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search_regressor.fit(X_train_reg_core, y_train_reg_core)

print("✅ Best Regressor Params:", grid_search_regressor.best_params_)

# 🧠 Final XGB Regressor (no evals, no early stopping)
best_params = grid_search_regressor.best_params_
xgb_reg = XGBRegressor(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    subsample=best_params['subsample'],
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
xgb_reg.fit(X_train_reg_core, y_train_reg_core)

# 📉 Validation set performance
y_val_pred = xgb_reg.predict(X_val_reg)
print("📊 Validation Set Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val_reg, y_val_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_val_reg, y_val_pred):.2f}")
print(f"R² Score: {r2_score(y_val_reg, y_val_pred):.2f}")

# 📊 Classification Results
y_pred_cls = xgb_cls.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test_cls_enc, y_pred_cls))

# 📉 Regression Results
y_pred_reg = xgb_reg.predict(X_test)
print(f"📈 RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")
print(f"📈 MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"📈 R² Score: {r2_score(y_test_reg, y_pred_reg):.2f}")

# 📌 Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_cls_enc, y_pred_cls), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Drug Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 📌 Feature Importance
importances = xgb_cls.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Top 20 Important Features (Classification)")
plt.bar(range(20), importances[indices][:20])
plt.xticks(range(20), X.columns[indices][:20], rotation=90)
plt.tight_layout()
plt.show()

# 📌 Dosage Prediction Error Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--')
plt.xlabel("Actual Dosage (mg)")
plt.ylabel("Predicted Dosage (mg)")
plt.title("Actual vs Predicted Dosage")
plt.grid(True)
plt.tight_layout()
plt.show()

# 💾 Save Models Locally
joblib.dump(xgb_cls, "xgb_classifier_local.pkl")
joblib.dump(xgb_reg, "xgb_regressor_tuned_local.pkl")

print("✅ Models and encoder saved successfully.")
modeltunedntb.py
Displaying modeltunedntb.py.
