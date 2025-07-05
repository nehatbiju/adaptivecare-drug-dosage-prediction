import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# Load models and encoder
xgb_cls = joblib.load("xgb_classifier_local.pkl")
xgb_reg = joblib.load("xgb_regressor_tuned_local.pkl")
le = joblib.load("labelfinal_encoder55final.pkl")

# Drug name mapping
label_to_drugname = {
    502: 'METFORMIN', 226: 'DICLOFENAC', 733: 'SIMVASTATIN', 496: 'MELOXICAM',
    797: 'TRAMADOL HCL', 88: 'ATORVASTATIN', 596: 'OMEPRAZOLE', 839: 'VITAMIN D',
    538: 'MIRTAZAPINE', 387: 'HYDROXYZ HCL', 147: 'CETIRIZINE HCL', 125: 'CEPHALEXIN',
    18: 'ACETAMINOPHEN', 31: 'ALBUTEROL', 33: 'ALENDRONATE NA', 36: 'ALL OTH HORM PREPAR',
    39: 'ALLERGY RELIEVERS NEC', 42: 'ALLOPURINOL', 49: 'ALPRAZOLAM', 74: 'AMOXICILLIN',
    86: 'ASPIRIN', 102: 'BENAZEPRIL HCL', 129: 'CHLORTHALIDONE', 131: 'CITALOPRAM HBR',
    155: 'CYCLOBENZAPRINE HCL', 157: 'DAPSONE', 174: 'DIVALPROEX SODIUM', 184: 'DOXEPIN HCL',
    185: 'DOXYCYCLINE HYCLATE', 186: 'DROSPIRENONE/ETHINYL', 201: 'ESCITALOPRAM OXALATE',
    235: 'FEXOFENADINE HCL', 263: 'FLUOXETINE HCL', 267: 'FLUTICASONE PROPIONATE',
    279: 'FUROSEMIDE', 285: 'GABAPENTIN', 294: 'GLIPIZIDE', 296: 'GLYBURIDE', 305: 'HCTZ',
    313: 'HYDROCHLOROTHIAZIDE', 327: 'HYDROCODONE/APAP', 331: 'HYDROXYCHLOROQUINE',
    336: 'IBUPROFEN', 340: 'INSULIN', 342: 'INSULIN GLARGINE', 349: 'ISOSORBIDE MONONITRATE',
    351: 'KETOCONAZOLE', 371: 'LEVOTHYROXINE SODIUM', 374: 'LISINOPRIL', 396: 'LOSARTAN POTASSIUM',
    409: 'MECLIZINE HCL', 410: 'MEDROXYPROGESTERONE', 428: 'METOPROLOL SUCCINATE',
    431: 'METOPROLOL TARTRATE', 434: 'MINOCYCLINE HCL', 447: 'NAPROXEN', 460: 'NIFEDIPINE',
    465: 'NITROGLYCERIN', 467: 'NITROFURANTOIN MONO-MACR', 469: 'NORTRIPTYLINE HCL',
    473: 'OLANZAPINE', 514: 'PREDNISONE', 524: 'PROPRANOLOL HCL', 525: 'QUETIAPINE FUMARATE',
    541: 'RANITIDINE HCL', 583: 'SERTRALINE HCL', 608: 'SULFAMETHOXAZOLE/TRIMETH',
    614: 'TAMSULOSIN HCL', 615: 'TEMAZEPAM', 624: 'TERAZOSIN HCL', 629: 'TIZANIDINE HCL',
    646: 'TRAZODONE HCL', 656: 'VALACYCLOVIR HCL', 662: 'VALPROIC ACID', 664: 'VENLAFAXINE HCL',
    668: 'VERAPAMIL HCL', 679: 'WARFARIN SODIUM', 692: 'ZIPRASIDONE HCL',
    714: 'ZOLPIDEM TARTRATE', 726: 'MULTIVITAMINS', 750: 'GLIMEPIRIDE',
    760: 'MONTELUKAST SODIUM', 766: 'VITAMIN C', 789: 'FLUCONAZOLE', 791: 'FOLIC ACID',
    799: 'GUAIFENESIN', 819: 'LORAZEPAM', 823: 'MAGNESIUM OXIDE', 844: 'VITAMIN B12',
    847: 'VITAMIN E', 849: 'VITAMIN K', 854: 'VITAMINS NEC', 861: 'ZINC OXIDE'
}

st.title("💊 Adaptive Drug and Dosage Prediction")
st.write("Enter patient health information below to predict a suitable drug and dosage.")

# Collect input features
input_fields = {
    "AGELAST": st.number_input("Age", 0, 120, 40),
    "SEX.1": st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male"),
    "HIBPDX": st.selectbox("Hypertension Diagnosis", [0, 1]),
    "CHDDX": st.selectbox("Coronary Heart Disease", [0, 1]),
    "CANCERDX": st.selectbox("Cancer Diagnosis", [0, 1]),
    "DIABDX_M18": st.selectbox("Diabetes Diagnosis", [0, 1]),
    "ARTHDX": st.selectbox("Arthritis Diagnosis", [0, 1]),
    "ASTHDX": st.selectbox("Asthma Diagnosis", [0, 1]),
    "CHPMED42": st.selectbox("Prescribed Cholesterol Meds", [0, 1]),
    "CHTHER42": st.selectbox("Cholesterol Therapy", [0, 1]),
    "SAQELIG": st.selectbox("Eligible for Survey", [0, 1]),
    "ADGENH42": st.slider("General Health Rating", 1, 5, 3),
    "ADSMOK42": st.selectbox("Current Smoker", [0, 1]),
    "ADNSMK42": st.selectbox("Non-smoker Ever", [0, 1]),
    "DSMED53": st.selectbox("Taking Medication", [0, 1]),
    "HIBPDX.1": st.selectbox("Hypertension (Repeat)", [0, 1]),
    "OHRTDX": st.selectbox("Other Heart Disease", [0, 1]),
    "CHOLDX": st.selectbox("Cholesterol Diagnosis", [0, 1]),
    "RACEV1X": st.selectbox("Race V1", [1, 2, 3]),
    "RACEV2X": st.selectbox("Race V2", [1, 2, 3]),
    "RACEAX": st.selectbox("Race A", [0, 1]),
    "RACEBX": st.selectbox("Race B", [0, 1]),
    "RACEWX": st.selectbox("Race White", [0, 1]),
    "RACETHX": st.selectbox("Race Ethnicity", [1, 2]),
    "HISPANX": st.selectbox("Hispanic Origin", [0, 1]),
    "RTHLTH31": st.slider("Perceived Health", 1, 5, 2),
    "MNHLTH31": st.slider("Mental Health", 1, 5, 3),
}

# Estimate weight
def estimate_weight(age, gender):
    return 2.5 * age + 15 if gender == 1 else 2.3 * age + 14

input_fields["WEIGHT_KG"] = estimate_weight(input_fields["AGELAST"], input_fields["SEX.1"])

# Predict
if st.button("🔍 Predict Drug and Dosage"):
    input_df = pd.DataFrame([input_fields])
    required_features = xgb_cls.get_booster().feature_names

    for col in required_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[required_features]

    # Predict drug
    encoded_pred = int(xgb_cls.predict(input_df)[0])
    drug_name = label_to_drugname.get(le.inverse_transform([encoded_pred])[0], "Unknown Drug")

    # Predict dosage
    dosage = xgb_reg.predict(input_df)[0]

    st.success(f"🧠 Predicted Drug: **{drug_name}**")
    st.info(f"💊 Recommended Dosage: **{dosage:.2f} mg**")
