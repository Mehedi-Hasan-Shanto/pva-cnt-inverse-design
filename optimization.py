import pandas as pd
import numpy as np
import re
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# === Step 1: Load and preprocess equations ===
df_eq = pd.read_csv("rsm_normalized_equations_final.csv")

def sanitize_equation(eqn):
    eqn = eqn.replace("Y_scaled = ", "").replace("^2", "**2")
    eqn = eqn.replace("chirality_diameter", "x0")
    eqn = eqn.replace("treat_untreated", "x1")
    eqn = eqn.replace("treat_hno3", "x2")
    eqn = eqn.replace("treat_pd", "x3")
    eqn = re.sub(r'(\bx\d\b)\s+(\bx\d\b)', r'\1*\2', eqn)
    return eqn

parsed_equations = [sanitize_equation(eq) for eq in df_eq["Normalized_Equation"]]

# === Step 2: Define Valid Combinations ===
valid_combinations = [
    [13.56, 1, 0, 0],
    [16.27, 1, 0, 0],
    [13.56, 0, 1, 0],
    [16.27, 0, 1, 0],
    [13.56, 0, 0, 1],
    [16.27, 0, 0, 1],
]
valid_combinations = np.array(valid_combinations)

# === Step 3: Define the Optimization Problem ===
class CompositeOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,
            n_obj=7,
            n_constr=0,
            xl=np.min(valid_combinations, axis=0),
            xu=np.max(valid_combinations, axis=0)
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Snap inputs to nearest valid combination
        snapped_X = np.array([valid_combinations[np.argmin(np.linalg.norm(valid_combinations - x, axis=1))] for x in X])

        x0, x1, x2, x3 = snapped_X[:, 0], snapped_X[:, 1], snapped_X[:, 2], snapped_X[:, 3]
        Y = []
        for expr in parsed_equations:
            Y.append(eval(expr, {"np": np}, {"x0": x0, "x1": x1, "x2": x2, "x3": x3}))
        Y = np.array(Y).T

        out["F"] = np.column_stack([
            -Y[:, 0], -Y[:, 1], -Y[:, 2], -Y[:, 3],  # Maximize
             Y[:, 4],  Y[:, 5], -Y[:, 6]            # MSD, Porosity: minimize; Tg: maximize
        ])
        out["X"] = snapped_X

# === Step 4: Run Optimization ===
algorithm = NSGA2(
    pop_size=150,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.8, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

result = minimize(
    CompositeOptimizationProblem(),
    algorithm,
    ('n_gen', 150),
    seed=1,
    verbose=True
)

# === Step 5: Extract and Save Results ===
X_opt = result.opt.get("X")
F_opt = result.opt.get("F")

feature_names = ["Modulus", "Strength", "Toughness", "Energy", "MSD", "Porosity", "Tg"]
output_data = []

for x, f in zip(X_opt, F_opt):
    result_dict = {
        "chirality_diameter": x[0],
        "treat_untreated": x[1],
        "treat_hno3": x[2],
        "treat_pd": x[3]
    }
    result_dict.update({
        f"{name}_scaled": (-val if name not in ["MSD", "Porosity"] else val)
        for name, val in zip(feature_names, f)
    })
    output_data.append(result_dict)

df_output = pd.DataFrame(output_data)
df_output.to_csv("optimized_composite_result_fixed.csv", index=False)

# === Step 6: Find and Print Best (Optimum) Solution ===
ideal_vector = np.min(F_opt, axis=0)
distances = np.linalg.norm(F_opt - ideal_vector, axis=1)
best_idx = np.argmin(distances)
best_solution = df_output.iloc[best_idx]

print("\n Optimum Composite Configuration:")
print(best_solution.to_string(index=True))

best_solution.to_frame().T.to_csv("optimum_composite_selected.csv", index=False)

# === Step 7: Parallel Coordinates Plot ===
df_plot = df_output.copy()
df_plot["Treatment"] = df_plot[["treat_untreated", "treat_hno3", "treat_pd"]].apply(
    lambda row: "untreated" if row[0] == 1 else "HNO3" if row[1] == 1 else "Pd", axis=1
)

plot_cols = [f"{name}_scaled" for name in feature_names]
df_plot_vis = df_plot[plot_cols].copy()
df_plot_vis["Treatment"] = df_plot["Treatment"]

plt.figure(figsize=(12, 6))
parallel_coordinates(df_plot_vis, class_column="Treatment", colormap=plt.cm.Set1, alpha=0.7)
plt.title("Parallel Coordinates Plot for Optimized Composite Responses")
plt.ylabel("Scaled Response Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
