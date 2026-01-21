import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dynesty import NestedSampler
from scipy.special import gammaln

# Models
def background_model(x, A, b):
    return A * np.exp(-b * (x - 100))

def signal_plus_background(x, A, b, mu, S, sigma):
    bg = background_model(x, A, b)
    sig = S * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return bg + sig

# Log-likelihood
def log_poisson(n, lam):
    lam = np.maximum(lam, 1e-12)
    return np.sum(n*np.log(lam) - lam - gammaln(n+1))

# Priors

def prior_B(u):
    A = 0.01 + 300 * u[0]    # amplitude ~ 10–210
    b = 0.01 + 4.0 * u[1]   # slope ~ 0.5–2.5
    return A, b

def prior_SB(u):
    A = 0.01 + 300 * u[0]          # background amplitude 10–210
    b = 0.01 + 4.0 * u[1]         # slope 0.5–2.5
    mu = 90 + 60 * u[2]         # signal mass 120–130 GeV
    # log-uniform signal amplitude 0.1–20
    logS = np.log10(0.1) + u[3]*(np.log10(20)-np.log10(0.1))
    S = 10**logS
    sigma = 0.01 + 3.0*u[4]       # width 0.5–2.5
    return A, b, mu, S, sigma


# Likelihood wrappers
def make_loglike_B(x, n):
    def loglike(theta):
        A, b = theta
        lam = background_model(x, A, b)
        return log_poisson(n, lam)
    return loglike

def make_loglike_SB(x, n):
    def loglike(theta):
        A, b, mu, S, sigma = theta
        lam = signal_plus_background(x, A, b, mu, S, sigma)
        return log_poisson(n, lam)
    return loglike

# Run Bayesian analysis
def run_bayes(file, maxiter_B=5000, maxiter_SB=10000, nlive_B=400, nlive_SB=600):
    df = pd.read_csv(file)
    x = df["mass"].values
    n = df["observed_total"].values

    # Background-only model
    sampler_B = NestedSampler(
        make_loglike_B(x, n),
        prior_B,
        ndim=2,
        nlive=nlive_B,
        sample='rwalk',
        bootstrap=0
    )
    sampler_B.run_nested(maxiter=maxiter_B)
    ZB = sampler_B.results.logz[-1]
    samples_B = sampler_B.results.samples

    # Signal + background model
    sampler_SB = NestedSampler(
        make_loglike_SB(x, n),
        prior_SB,
        ndim=5,
        nlive=nlive_SB,
        sample='rwalk',
        bootstrap=0
    )
    sampler_SB.run_nested(maxiter=maxiter_SB)
    ZSB = sampler_SB.results.logz[-1]
    samples_SB = sampler_SB.results.samples

    # Numerically stable posterior probability of background-only
    delta_logZ = ZSB - ZB
    if delta_logZ > 700:        # extremely strong signal
        posterior_prob_B = 0.0
    elif delta_logZ < -700:     # extremely strong background
        posterior_prob_B = 1.0
    else:
        posterior_prob_B = 1.0 / (1.0 + np.exp(delta_logZ))

    return {
        "logZ_B": ZB,
        "logZ_SB": ZSB,
        "posterior_p(B)": posterior_prob_B,
        "x": x,
        "n": n,
        "samples_B": samples_B,
        "samples_SB": samples_SB
    }

# Plotting
def plot_results(result):
    x = result["x"]
    n = result["n"]

    # Fitted background (median)
    A_median, b_median = np.median(result["samples_B"], axis=0)
    bg_fit = background_model(x, A_median, b_median)

    # Fitted signal+background (median)
    A_m, b_m, mu_m, S_m, sigma_m = np.median(result["samples_SB"], axis=0)
    sb_fit = signal_plus_background(x, A_m, b_m, mu_m, S_m, sigma_m)

    plt.figure(figsize=(10,6))
    plt.errorbar(x, n, yerr=np.sqrt(n), fmt='o', label='Observed')
    plt.plot(x, bg_fit, 'r--', label='Background fit (median)')
    plt.plot(x, sb_fit, 'g-', label='Signal+Background fit (median)')
    plt.xlabel("Mass")
    plt.ylabel("Counts")
    plt.title("Bayesian Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Posterior distributions
    samples_B = result["samples_B"]
    samples_SB = result["samples_SB"]

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.hist(samples_B[:,0], bins=30, alpha=0.6)
    plt.xlabel("A (background amplitude)")
    plt.ylabel("Posterior counts")
    plt.title("Background-only posterior")
    plt.subplot(1,2,2)
    plt.hist(samples_B[:,1], bins=30, alpha=0.6)
    plt.xlabel("b (background slope)")
    plt.ylabel("Posterior counts")
    plt.title("Background-only posterior")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    labels = ["A","b","mu","S","sigma"]
    for i in range(5):
        plt.subplot(2,3,i+1)
        plt.hist(samples_SB[:,i], bins=30, alpha=0.6)
        plt.xlabel(labels[i])
        plt.ylabel("Posterior counts")
        plt.title(f"Signal+Background posterior: {labels[i]}")
    plt.tight_layout()
    plt.show()

# CLI
if __name__ == "__main__":
    filename = input("Enter the CSV filename to analyze (e.g., dataset_L1.0.csv): ").strip()
    try:
        result = run_bayes(filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        exit(1)

    print(f"LogZ B: {result['logZ_B']:.2f}")
    print(f"LogZ SB: {result['logZ_SB']:.2f}")
    print(f"Posterior P(B|data): {result['posterior_p(B)']:.3e}")

    plot_results(result)
