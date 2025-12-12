# comparison.py

import pandas as pd
from frequentist import fit_background, q_local_for_mass, local_p_value_from_q, global_p_value_mc
from bayesian import run_bayes

def compare(file):
    # Fit the background once
    coeffs_bg = fit_background()

    # Load CSV data
    df = pd.read_csv(file)
    L = df["luminosity"].iloc[0]
    obs = df["observed_total"].values

    # Frequentist analysis
    q_local = q_local_for_mass(obs, L, coeffs_bg)
    local_p, local_Z = local_p_value_from_q(q_local)

    global_p, global_Z, _, _ = global_p_value_mc(obs, L, coeffs_bg, n_mc=300)

    # Bayesian analysis
    bayes = run_bayes(file)

    # Combine results
    return {
        "file": file,
        "local_Z": local_Z,
        "global_Z": global_Z,
        "logZ_B": bayes["logZ_B"],
        "logZ_SB": bayes["logZ_SB"],
        "posterior_p(B)": bayes["posterior_p(B)"]
    }

if __name__ == "__main__":
    file = input("Enter CSV filename to analyze: ").strip()
    result = compare(file)
    print(result)
