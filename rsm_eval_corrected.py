import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# Load the corrected dataset
df = pd.read_csv('predicted and actual.csv')

# Define the mapping of actual and predicted columns
features = {
    "Modulus": ("Modulus_actual", "Modulus_predicted"),
    "Tensile Strength": ("Tensile_strength_actual", "Tensile_strength_predicted"),
    "Toughness": ("Toughness_actual", "Toughness_predicted"),
    "Energy": ("Energy_actual", "Energy_predicted"),
    "MSD": ("MSD_actual", "MSD_predicted"),
    "Porosity": ("Porosity_actual", "Porosity_predicted"),
    "Tg": ("Tg_actual", "Tg_predicted"),
}

# Calculate R-squared, Adjusted R-squared, and MSE for each feature
results = []
n = len(df)
p = 2  # number of predictors: chirality and treatment
for feature, (actual_col, pred_col) in features.items():
    y_true = df[actual_col]
    y_pred = df[pred_col]
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    results.append({
        "Feature": feature,
        "R-squared": r2,
        "Adjusted R-squared": adj_r2,
        "MSE": mse
    })

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("rsm_model_evaluation_metrics.csv", index=False)
print("✅ Metrics saved to 'rsm_model_evaluation_metrics.csv'")
