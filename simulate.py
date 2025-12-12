import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# -----------------------------
# User-defined scales
# -----------------------------
BG_SCALE = 2000   # approximate total background counts at L=1
SIG_SCALE = 300  # peak signal counts at target luminosity (L ~ 5 fb)

# -----------------------------
# Mass window + binning
# -----------------------------
M_MIN = 100
M_MAX = 160
BIN_WIDTH = 2
BIN_EDGES = np.arange(M_MIN, M_MAX + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

# -----------------------------
# Bernstein background (4th order)
# -----------------------------
def bernstein_4(m, c0, c1, c2, c3, c4):
    t = (m - M_MIN) / (M_MAX - M_MIN)
    b0 = (1 - t)**4
    b1 = 4 * t * (1 - t)**3
    b2 = 6 * t**2 * (1 - t)**2
    b3 = 4 * t**3 * (1 - t)
    b4 = t**4
    return c0*b0 + c1*b1 + c2*b2 + c3*b3 + c4*b4

def fit_background(initial_coeffs=[0.0045, 0.0025, 0.00075, 0.0004, 0.0001], seed=999):
    """
    Fit Bernstein coefficients for background.
    initial_coeffs are normalized and multiplied by BG_SCALE.
    """
    initial_coeffs = np.array(initial_coeffs) * BG_SCALE
    rng = np.random.default_rng(seed)
    rough = bernstein_4(BIN_CENTERS, *initial_coeffs)
    obs = rng.poisson(rough)
    coeffs, _ = curve_fit(bernstein_4, BIN_CENTERS, obs, p0=initial_coeffs, bounds=(0, np.inf))
    return coeffs

# -----------------------------
# Gaussian Higgs-like signal
# -----------------------------
def gaussian_signal(m, m_h=125.0, sigma=1.5, amplitude=SIG_SCALE):
    return amplitude * np.exp(-(m - m_h)**2 / (2*sigma**2))

# -----------------------------
# Luminosity scaling + Poisson sampling
# -----------------------------
def expected_counts(L, B, S):
    B_L = L * B
    S_L = L * S
    return B_L, S_L, B_L + S_L

def sample_poisson(B_L, S_L, TOT_L, seed=None):
    rng = np.random.default_rng(seed)
    obs_B = rng.poisson(B_L)
    obs_TOT = rng.poisson(TOT_L)
    return obs_B, obs_TOT

# -----------------------------
# CSV export
# -----------------------------
def save_csv(bin_centers, B_L, S_L, TOT_L, obs_B, obs_TOT, L, fname):
    df = pd.DataFrame({
        "mass": bin_centers,
        "expected_background": B_L,
        "expected_signal": S_L,
        "expected_total": TOT_L,
        "observed_background": obs_B,
        "observed_total": obs_TOT,
        "luminosity": L
    })
    df.to_csv(fname, index=False)
    print(f"Saved: {fname}  (sum expected_total={TOT_L.sum():.1f}, observed_total={obs_TOT.sum():.0f})")
    return df

# -----------------------------
# Single luminosity simulation
# -----------------------------
def simulate_luminosity(L, coeffs_bg, signal_params=(125.0, 1.5, SIG_SCALE), seed=None, fname=None):
    """
    Simulate dataset for a single luminosity L.
    Signal is scaled to be hidden at low L and detectable around L ~ 5 fb.
    """
    m_h, sigma, amplitude_max = signal_params
    B = bernstein_4(BIN_CENTERS, *coeffs_bg)

    # Signal amplitude scales slowly with luminosity
    scale_L0 = 5.0   # target detection luminosity
    amplitude_scaled = amplitude_max * (L / scale_L0)**(1/3.5)  # slow growth

    S = gaussian_signal(BIN_CENTERS, m_h, sigma, amplitude_scaled)
    B_L, S_L, TOT_L = expected_counts(L, B, S)
    obs_B, obs_TOT = sample_poisson(B_L, S_L, TOT_L, seed)

    if fname is not None:
        save_csv(BIN_CENTERS, B_L, S_L, TOT_L, obs_B, obs_TOT, L, fname)

    return B_L, S_L, TOT_L, obs_B, obs_TOT

# -----------------------------
# Batch generation
# -----------------------------
def generate_all_luminosities(luminosities, signal_params=(125.0, 1.5, SIG_SCALE), seed=1234):
    coeffs_bg = fit_background()
    print("Fitted background coefficients:", np.round(coeffs_bg, 3))
    total_B_at_L1 = bernstein_4(BIN_CENTERS, *coeffs_bg).sum()
    print(f"Total expected background at L=1: {total_B_at_L1:.1f} (target ~{BG_SCALE})")

    for L in luminosities:
        fname = f"dataset_L{L}.csv"
        simulate_luminosity(L, coeffs_bg, signal_params, seed, fname)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    Ls = input("Luminosities (comma-separated, e.g., 0.1,1,5): ").strip()
    luminosities = [float(L.strip()) for L in Ls.split(",") if L.strip() != ""]

    amp = float(input(f"Signal amplitude at L=5 fb (default {SIG_SCALE}): ").strip() or SIG_SCALE)
    mh = float(input("Signal mass m_h (default 125): ").strip() or 125.0)
    sigma = float(input("Signal width sigma (default 1.5): ").strip() or 1.5)
    seed = int(input("Random seed (default 1234): ").strip() or 1234)

    print(f"Simulating luminosities: {luminosities}, signal amp(L=5 fb)={amp}")
    generate_all_luminosities(luminosities, signal_params=(mh, sigma, amp), seed=seed)
