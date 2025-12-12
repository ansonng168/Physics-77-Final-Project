import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar, curve_fit

M_MIN = 100
M_MAX = 160
BIN_WIDTH = 2
BIN_EDGES = np.arange(M_MIN, M_MAX + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
EPS = 1e-12

# Bernstein background
def bernstein_4(m, c0, c1, c2, c3, c4):
    t = (m - M_MIN) / (M_MAX - M_MIN)
    b0 = (1 - t)**4
    b1 = 4 * t * (1 - t)**3
    b2 = 6 * t**2 * (1 - t)**2
    b3 = 4 * t**3 * (1 - t)
    b4 = t**4
    return c0*b0 + c1*b1 + c2*b2 + c3*b3 + c4*b4

def fit_background(initial_coeffs=[4, 2, 0.5, 0.15, 0.05], seed=999):
    rng = np.random.default_rng(seed)
    rough = bernstein_4(BIN_CENTERS, *initial_coeffs)
    obs = rng.poisson(rough)
    coeffs, _ = curve_fit(bernstein_4, BIN_CENTERS, obs, p0=initial_coeffs, bounds=(0, np.inf))
    return coeffs

# Gaussian signal
def gaussian_signal(m, m_h=125.0, sigma=1.5, amplitude=2.5):
    return amplitude * np.exp(-(m - m_h)**2 / (2 * sigma**2))

# Frequentist calculations
def poisson_nll(lam, n):
    lam = np.asarray(lam)
    n = np.asarray(n)
    return np.sum(lam - n * np.log(lam + EPS))

def expected_background(L, coeffs_bg):
    return L * bernstein_4(BIN_CENTERS, *coeffs_bg)

def expected_signal_shape(m_h, sigma=1.5):
    return gaussian_signal(BIN_CENTERS, m_h, sigma, amplitude=1.0)

def expected_signal_background(L, coeffs_bg, mu, m_h, sigma=1.5):
    lam_B = expected_background(L, coeffs_bg)
    S_unit = expected_signal_shape(m_h, sigma)
    return lam_B + L * mu * S_unit

def q_local_for_mass(obs_counts, L, coeffs_bg, m_h=125.0, sigma=1.5, mu_max=10.0):
    lam_B = expected_background(L, coeffs_bg)
    nll_b = poisson_nll(lam_B, obs_counts)
    def nll_sb(mu):
        if mu < 0: return np.inf
        lam_SB = expected_signal_background(L, coeffs_bg, mu, m_h, sigma)
        return poisson_nll(lam_SB, obs_counts)
    res = minimize_scalar(nll_sb, bounds=(0, mu_max), method="bounded")
    return max(0, 2 * (nll_b - res.fun))

def local_p_value_from_q(q):
    Z = np.sqrt(max(q, 0))
    p = 1 - norm.cdf(Z)
    return p, Z

def q_global_max_for_dataset(obs_counts, L, coeffs_bg, mh_grid=None):
    if mh_grid is None: mh_grid = np.arange(110, 150, 1)
    q_vals = [q_local_for_mass(obs_counts, L, coeffs_bg, m_h) for m_h in mh_grid]
    return max(q_vals), np.array(q_vals), mh_grid

def simulate_dataset(L, coeffs_bg, with_signal, signal_params, seed=None):
    m_h, sigma, amp = signal_params
    if not with_signal: amp = 0.0
    B = bernstein_4(BIN_CENTERS, *coeffs_bg)
    S = gaussian_signal(BIN_CENTERS, m_h, sigma, amp)
    TOT_L = L * (B + S)
    rng = np.random.default_rng(seed)
    return rng.poisson(TOT_L)

def global_p_value_mc(obs_counts, L, coeffs_bg, mh_grid=None, n_mc=500, signal_params=(125,1.5,2.5), rng_seed=123):
    q_obs_max, _, _ = q_global_max_for_dataset(obs_counts, L, coeffs_bg, mh_grid)
    rng = np.random.default_rng(rng_seed)
    q_samples = [q_global_max_for_dataset(simulate_dataset(L, coeffs_bg, False, signal_params, rng.integers(0,2**32-1)), L, coeffs_bg, mh_grid)[0] for _ in range(n_mc)]
    p_global = min(max(np.mean(np.array(q_samples) >= q_obs_max), 1e-12), 1-1e-12)
    Z_global = norm.isf(p_global)
    return p_global, Z_global, q_obs_max, q_samples

def run_on_csv(filename, coeffs_bg):
    df = pd.read_csv(filename)
    L = df["luminosity"].iloc[0]
    obs = df["observed_total"].values
    q_local = q_local_for_mass(obs, L, coeffs_bg)
    p_local, Z_local = local_p_value_from_q(q_local)
    p_global, Z_global, _, _ = global_p_value_mc(obs, L, coeffs_bg, n_mc=300)
    print("\n===== Frequentist Results =====")
    print(f"File: {filename}")
    print(f"Luminosity: {L} fb^-1")
    print("Local q:", q_local)
    print("Local p-value:", p_local)
    print("Local Z:", Z_local)
    print("Global p-value:", p_global)
    print("Global Z:", Z_global)

# Main
if __name__ == "__main__":
    csv_file = input("Enter CSV filename to analyze: ").strip()
    print("\nFitting background model once...")
    coeffs_bg = fit_background()
    print("Background coefficients:", coeffs_bg)
    run_on_csv(csv_file, coeffs_bg)
