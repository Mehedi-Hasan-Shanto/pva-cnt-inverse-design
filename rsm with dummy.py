import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import statsmodels.api as sm
import joblib
import os

# === Step 1: Load Updated Original Values from CSV ===

df_orig = pd.read_csv("original values.csv")

# === Step 2: Map Chirality to Diameter (numeric input) ===
chirality_map = {"(10,10)": 13.56, "(12,12)": 16.27}
df_orig["chirality_diameter"] = df_orig["Chirality"].map(chirality_map)

# === Step 3: Manually Encode Treatment as Dummy Variables (No drop) ===
treatment_mapping = {
    "untreated": [1, 0, 0],
    "hno3": [0, 1, 0],
    "pd": [0, 0, 1]
}
df_orig["Treatment"] = df_orig["Treatment"].str.lower()
df_orig[["treat_untreated", "treat_hno3", "treat_pd"]] = df_orig["Treatment"].map(treatment_mapping).apply(pd.Series)

# === Step 4: Normalize response columns ===
response_cols = [col for col in df_orig.columns if col not in ['Chirality', 'Treatment', 'chirality_diameter', 'treat_untreated', 'treat_hno3', 'treat_pd']]

def sanitize_filename(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("³", "3").replace("²", "2").replace("?", "").replace(".", "").replace(",", "")

scalers = {}
for col in response_cols:
    scaler = MinMaxScaler()
    df_orig[col + '_scaled'] = scaler.fit_transform(df_orig[[col]])
    scalers[col] = scaler
    safe_name = sanitize_filename(col)
    joblib.dump(scaler, f"{safe_name}.pkl")

# === Step 5: Fit RSM Models ===
feature_names = ['chirality_diameter', 'treat_untreated', 'treat_hno3', 'treat_pd']
X = df_orig[feature_names]
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly_array = poly.fit_transform(X)
X_poly = pd.DataFrame(X_poly_array, columns=poly.get_feature_names_out(feature_names))

equations = {}
coeff_df = pd.DataFrame()
eq_list = []

for col in response_cols:
    y = df_orig[col + '_scaled']
    model = sm.OLS(y, X_poly).fit()
    params = model.params

    coeff_df[col] = params

    terms = [f"{params[i]:+.4f}*{X_poly.columns[i]}" for i in range(len(params))]
    equation = "Y_scaled = " + " ".join(terms).replace('+', '+ ').replace('-', '- ')
    equations[col] = equation
    eq_list.append({'Property': col, 'Normalized_Equation': equation})

# === Step 6: Save Results ===
eq_df = pd.DataFrame(eq_list)
eq_df.to_csv("rsm_normalized_equations_final.csv", index=False)
coeff_df.index.name = "Term"
coeff_df.to_csv("rsm_coefficients_scaled_final.csv")

eq_df.to_csv("rsm_normalized_equations_final.csv", index=False)
print("✅ Equations saved to rsm_normalized_equations_final.csv")

