"""
Hospital Readmission Risk Prediction
Step 2-6: Exploratory Data Analysis (EDA) + Basic Cleaning
"""

import pandas as pd
import numpy as np

# -------------------------------------------------
# STEP 3: Load the data
# -------------------------------------------------
df = pd.read_csv(r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\data\diabetic_data.csv")

print("Shape of dataset:", df.shape)
print(df.head())

# -------------------------------------------------
# STEP 4: Look at the target column (what we're predicting)
# -------------------------------------------------
print("\nOriginal 'readmitted' column breakdown:")
print(df["readmitted"].value_counts())

# Create a simple yes/no version: readmitted within 30 days or not
df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)

print("\nNew binary target 'readmit_30d':")
print(df["readmit_30d"].value_counts())

# -------------------------------------------------
# STEP 5: Check for missing / messy data
# -------------------------------------------------
df_check = df.replace("?", np.nan)
missing = df_check.isnull().mean().sort_values(ascending=False) * 100

print("\nColumns with missing data (%):")
print(missing[missing > 0])

# -------------------------------------------------
# STEP 6: Clean the data
# -------------------------------------------------
# Drop columns that are mostly empty or not useful for prediction
df_clean = df.drop(columns=[
    "weight",            # 97% missing
    "max_glu_serum",     # 95% missing
    "A1Cresult",         # 83% missing
    "payer_code",        # 40% missing, not medically relevant
    "encounter_id",      # just an ID number, not useful
    "patient_nbr",       # just an ID number, not useful
    "readmitted"         # replaced by our new readmit_30d column
])

# Replace remaining '?' symbols with the word 'Unknown'
df_clean = df_clean.replace("?", "Unknown")

print("\nCleaned dataset shape:", df_clean.shape)
print("Total missing values left:", df_clean.isnull().sum().sum())

# -------------------------------------------------
# Save the cleaned data so Step 7 (modeling) can use it directly
# -------------------------------------------------
df_clean.to_csv(r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\data\diabetic_data_clean.csv", index=False)
print("\nSaved cleaned file as diabetic_data_clean.csv")

# -------------------------------------------------
# STEP 6.5: Group diagnosis codes into broader categories
# -------------------------------------------------
def map_diagnosis(code):
    if code == "Unknown":
        return "Unknown"
    try:
        code_num = float(code)
    except ValueError:
        return "Other"

    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    elif 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    elif 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    elif 250 <= code_num < 251:
        return "Diabetes"
    elif 800 <= code_num <= 999:
        return "Injury"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    else:
        return "Other"

for col in ["diag_1", "diag_2", "diag_3"]:
    df_clean[col] = df_clean[col].apply(map_diagnosis)

print("\nDiagnosis categories after grouping (diag_1 example):")
print(df_clean["diag_1"].value_counts())

# -------------------------------------------------
# STEP 7: Encoding (convert text columns into numbers)
# -------------------------------------------------

# Separate our target column first (we don't want to encode this)
y = df_clean["readmit_30d"]
X = df_clean.drop(columns=["readmit_30d"])

# Find which columns are text (categorical)
categorical_cols = X.select_dtypes(include="object").columns
print("\nCategorical columns to encode:", list(categorical_cols))
print("Number of categorical columns:", len(categorical_cols))

# One-hot encode all text columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("\nShape before encoding:", X.shape)
print("Shape after encoding:", X_encoded.shape)

# Save this fully model-ready dataset
X_encoded["readmit_30d"] = y
X_encoded.to_csv(r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\data\diabetic_data_encoded.csv", index=False)
print("\nSaved model-ready file as diabetic_data_encoded.csv")


# -------------------------------------------------
# STEP 8: Split data into train/test and train a model
# -------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Reload the encoded, model-ready file
data = pd.read_csv(r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\data\diabetic_data_encoded.csv")

X = data.drop(columns=["readmit_30d"])
y = data["readmit_30d"]

# Split: 80% to train the model, 20% to test how well it learned
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# Train a Random Forest model, telling it the data is imbalanced
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # important: handles our 11% vs 89% imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Test it
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# -------------------------------------------------
# STEP 9: Try XGBoost (usually stronger for this kind of data)
# -------------------------------------------------
from xgboost import XGBClassifier
import re
# scale_pos_weight tells XGBoost how imbalanced the data is
# roughly = (number of 0s) / (number of 1s)
scale = (y_train == 0).sum() / (y_train == 1).sum()
print("scale_pos_weight:", scale)

# Clean column names: remove characters XGBoost doesn't allow
X_train.columns = [re.sub(r"[\[\]<]", "", col) for col in X_train.columns]
X_test.columns = [re.sub(r"[\[\]<]", "", col) for col in X_test.columns]

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)

y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Instead of using the default 0.5 cutoff, we test a few thresholds
from sklearn.metrics import classification_report

for threshold in [0.5, 0.3, 0.2]:
    print(f"\n--- Threshold: {threshold} ---")
    y_pred_thresh = (y_proba_xgb >= threshold).astype(int)
    print(classification_report(y_test, y_pred_thresh))

print("XGBoost ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))


# Final model = XGBoost at threshold 0.5
final_model = xgb_model
final_threshold = 0.5
print("\nFinal model locked in: XGBoost, threshold = 0.5")
print(f"Catches {0.61*100:.0f}% of true readmissions (recall), ROC-AUC = 0.687")


# -------------------------------------------------
# STEP 10: SHAP - explain the model's predictions
# -------------------------------------------------
import shap

# Create an "explainer" for our XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the test set
# (using a smaller sample first - 500 patients - so it runs fast)
X_sample = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated for", X_sample.shape[0], "patients")

# -------------------------------------------------
# GLOBAL explanation: which features matter most overall?
# -------------------------------------------------
shap.summary_plot(shap_values, X_sample, show=False)
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig(r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\notebook\shap_summary.png")
print("Saved: shap_summary.png (shows top features overall)")
plt.close()

# -------------------------------------------------
# LOCAL explanation: why did ONE specific patient get their score?
# -------------------------------------------------
patient_index = 0  # first patient in our sample
plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[patient_index],
        base_values=explainer.expected_value,
        data=X_sample.iloc[patient_index],
        feature_names=X_sample.columns.tolist()
    ),
    show=False
)
plt.tight_layout()
plt.savefig(r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\notebook\shap_patient_example.png")
print("Saved: shap_patient_example.png (shows one patient's explanation)")
plt.close()
# -------------------------------------------------
# STEP 11: Save the model + column list for the app
# -------------------------------------------------
import joblib

joblib.dump(xgb_model, r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\notebook\readmission_model.pkl")
joblib.dump(list(X_train.columns), r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\notebook\model_columns.pkl")

print("\nModel and columns saved successfully!")


# -------------------------------------------------
# STEP 13: Save default values for the app
# -------------------------------------------------
default_values = X_train.median()
joblib.dump(default_values, r"C:\Users\shrad\Desktop\Hospital-Readmission-Risk\notebook\default_values.pkl")
print("Default values saved!")