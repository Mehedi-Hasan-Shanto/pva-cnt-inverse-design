import pandas as pd
import re

# === Step 1: Load input files ===
eq_df = pd.read_csv("rsm_normalized_equations_final.csv")
input_df = pd.read_csv("input value.csv")

# === Step 2: Helper function to fix missing multiplications ===
def insert_multiplication_operators(expr: str) -> str:
    # Insert * between adjacent variables or values (e.g., 'x y' → 'x*y')
    return re.sub(r'(?<=[0-9a-zA-Z_]) (?=[0-9a-zA-Z_(])', '*', expr)

# === Step 3: Evaluate equation for one input row ===
def final_evaluate_equation(equation: str, input_row: pd.Series) -> float:
    expr = equation.split("=", 1)[1].strip()
    expr = expr.replace("^", "**")                     # Replace ^ with ** for Python
    expr = insert_multiplication_operators(expr)       # Fix missing multiplications

    # Replace variable names with actual values
    for var in sorted(input_row.index, key=len, reverse=True):
        expr = re.sub(rf'\b{re.escape(var)}\b', str(input_row[var]), expr)

    return eval(expr)

# === Step 4: Apply all equations to the input data ===
results_df = input_df.copy()
for _, row in eq_df.iterrows():
    property_name = row["Property"]
    equation = row["Normalized_Equation"]
    results_df[property_name + " (Normalized)"] = input_df.apply(
        lambda x: final_evaluate_equation(equation, x), axis=1
    )

# === Step 5: Save the output ===
results_df.to_csv("normalized_predictions_output.csv", index=False)
print("Predictions saved to normalized_predictions_output.csv")
