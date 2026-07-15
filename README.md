Here's your complete, updated README with the screenshots section included:

```markdown
# Hospital Readmission Risk Predictor

Predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days of discharge, and explains *why* using SHAP — so predictions are interpretable, not just accurate.

## Problem

Hospital readmissions within 30 days are costly and often preventable. Identifying high-risk patients at discharge allows hospitals to intervene early (follow-up calls, care plans) rather than reactively. This project builds a model to flag at-risk patients and explains the reasoning behind each prediction, since a risk score alone isn't actionable for clinicians without knowing *why*.

## Dataset

**Diabetes 130-US Hospitals (1999–2008)** — 101,766 real, de-identified patient encounters across 130 US hospitals, with 50 features covering demographics, admission details, diagnoses, medications, and lab results.
Source: [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes)

## Approach

1. **EDA** — analyzed target distribution, identified severe class imbalance (~11% readmitted within 30 days) and high-missingness columns
2. **Cleaning** — dropped columns with >35% missing data, grouped 700+ diagnosis codes into 9 clinical categories
3. **Encoding** — one-hot encoded categorical features (176 final features)
4. **Modeling** — trained Random Forest (baseline) and XGBoost with `scale_pos_weight` to address class imbalance
5. **Explainability** — applied SHAP (TreeExplainer) for global feature importance and per-patient prediction breakdowns
6. **Deployment** — built an interactive, hospital-themed Streamlit app for live risk scoring with explanations

## Results

| Model | ROC-AUC | Recall (readmitted class) |
|---|---|---|
| Random Forest (baseline) | 0.656 | 0.00 |
| **XGBoost (final)** | **0.687** | **0.61** |

The baseline model, despite balanced class weighting, failed to identify readmitted patients (0% recall) — it defaulted to predicting the majority class. Switching to XGBoost with `scale_pos_weight` improved recall to 61%, meaning the model correctly flags 6 out of 10 patients who are actually readmitted within 30 days — a meaningful, honest result on real imbalanced clinical data.

**Key drivers of readmission risk (via SHAP):** number of prior inpatient admissions, discharge disposition type, and number of diagnoses recorded were the strongest predictors — consistent with clinical intuition that recent hospital history predicts future hospital use.

## App Screenshots

**Low Risk Patient**
![Low Risk](snapshot/low_risk_example.png)

**High Risk Patient**
![High Risk](snapshot/high_risk_example.png)

## Project Structure

```
hospital-readmission-risk/
├── .streamlit/
│   └── config.toml                # app theme colors
├── data/
│   ├── diabetic_data.csv          # raw dataset
│   └── diabetic_data_clean.csv    # cleaned dataset
├── notebook/
│   ├── eda.py                     # EDA, cleaning, modeling, SHAP
│   ├── readmission_model.pkl      # saved trained model
│   ├── model_columns.pkl          # saved feature list
│   ├── default_values.pkl         # saved default patient values
│   ├── shap_summary.png           # global feature importance
│   └── shap_patient_example.png   # single-patient explanation
├── app/
│   └── app.py                     # Streamlit app
├── snapshot/                      # app output screenshots
├── requirements.txt
└── README.md
```

## How to Run

**1. Clone the repository**
```
git clone <your-repo-url>
cd hospital-readmission-risk
```

**2. Create and activate a virtual environment**

Windows:
```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```
pip install -r requirements.txt
```

**4. Run the full pipeline** (cleans data, trains model, generates SHAP plots, saves model files)
```
python notebook/eda.py
```

**5. Launch the interactive app**
```
streamlit run app/app.py
```
This opens a browser tab where you can input patient details and get a live risk score with a SHAP explanation.


## Tools Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib

## Future Improvements

- Deploy app publicly via Streamlit Community Cloud
