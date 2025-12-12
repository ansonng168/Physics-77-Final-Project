import glob
import pandas as pd
import numpy as np
from comparison import compare  # import the compare function

# -----------------------------
# Find all CSV files
# -----------------------------
csv_files = sorted(glob.glob("dataset_L*.csv"))  # all dataset_L*.csv
results = []

for file in csv_files:
    print(f"Processing {file} ...")
    
    # Run the comparison function
    result = compare(file)
    
    # Extract luminosity from filename, e.g., dataset_L1.0.csv -> 1.0
    lum_str = file.split("_L")[1].replace(".csv", "")
    luminosity = float(lum_str)
    
    results.append({
        "luminosity_fb": luminosity,
        "local_Z": float(result.get("local_Z", np.nan)),
        "global_Z": float(result.get("global_Z", np.nan)),
        "logZ_B": float(result.get("logZ_B", np.nan)),
        "logZ_SB": float(result.get("logZ_SB", np.nan)),
        "posterior_pB": float(result.get("posterior_p(B)", np.nan))
    })

# -----------------------------
# Create DataFrame
# -----------------------------
df_summary = pd.DataFrame(results)
df_summary = df_summary.sort_values("luminosity_fb").reset_index(drop=True)

# -----------------------------
# Save to CSV
# -----------------------------
df_summary.to_csv("summary_results.csv", index=False)
print("All results saved to summary_results.csv")
print(df_summary)
