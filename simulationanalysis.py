import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

m_min = 100
m_max = 160
bin_width = 2

bin_edges = np.arange(m_min, m_max + bin_width, bin_width)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

def bernstein_4(m, c0, c1, c2, c3, c4):
    t = (m - m_min) / (m_max - m_min)
    b0 = (1 - t)**4
    b1 = 4 * t * (1 - t)**3
    b2 = 6 * t**2 * (1 - t)**2
    b3 = 4 * t**3 * (1 - t)
    b4 = t**4
    return c0*b0 + c1*b1 + c2*b2 + c3*b3 + c4*b4

initial_guess = [4, 2, 0.5, 0.15, 0.05]

BG_SCALE = 2000
SIG_SCALE = 300

B_background_guess = bernstein_4(bin_centers, *initial_guess)

rng_fit = np.random.default_rng(seed=999)
obs_bg_for_fit = rng_fit.poisson(B_background_guess)

coeffs_fit, cov = curve_fit(
    bernstein_4,
    bin_centers,
    obs_bg_for_fit,
    p0=initial_guess,
    bounds=(0, np.inf)
)

B_background = BG_SCALE * bernstein_4(bin_centers, *coeffs_fit)

def smooth_background_curve(m_vals):
    return BG_SCALE * bernstein_4(m_vals, *coeffs_fit)

m_h = 125.0
sigma = 1.5
A_signal = 2.0

S_signal = SIG_SCALE * A_signal * np.exp(
    -(bin_centers - m_h)**2 / (2 * sigma**2)
)

def smooth_signal_curve(m_vals, m_h, sigma, A_signal):
    return SIG_SCALE * A_signal * np.exp(
        -(m_vals - m_h)**2 / (2 * sigma**2)
    )

lambda_background = B_background
lambda_signal_plus_background = B_background + S_signal

rng = np.random.default_rng(seed=42)

obs_background = rng.poisson(lambda_background)
obs_signal_plus_background = rng.poisson(lambda_signal_plus_background)

def scale_with_luminosity(L, background, signal):
    bg_L = L * background
    sig_L = L * signal
    return bg_L, sig_L, bg_L + sig_L

def poisson_sample(bg_L, tot_L, rng):
    obs_bg = rng.poisson(bg_L)
    obs_tot = rng.poisson(tot_L)
    return obs_bg, obs_tot

def plot_spectrum_subplot(ax, bin_centers, bg_L, sig_L, tot_L, obs_bg, obs_tot, L):
    m_fine = np.linspace(m_min, m_max, 2000)

    bg_smooth = L * smooth_background_curve(m_fine)
    sig_smooth = L * smooth_signal_curve(m_fine, m_h, sigma, A_signal)
    tot_smooth = bg_smooth + sig_smooth

    ax.plot(m_fine, bg_smooth, linewidth=2, label='Background (expected)')
    ax.plot(m_fine, tot_smooth, linewidth=2, label='S+B (expected)')

    ax.scatter(bin_centers, obs_bg, s=35, alpha=0.7)
    ax.scatter(bin_centers, obs_tot, s=35, alpha=0.7)

    ax.set_title(f"L = {L} fb$^{{-1}}$")
    ax.set_xlabel("m$_{\gamma\gamma}$ (GeV)")
    ax.set_ylabel("Counts / 2 GeV")
    ax.grid(alpha=0.2)

luminosities = [0.01, 0.5, 1, 5, 25]
rng = np.random.default_rng(seed=123)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, L in enumerate(luminosities):
    bg_L, sig_L, tot_L = scale_with_luminosity(L, B_background, S_signal)
    obs_bg, obs_tot = poisson_sample(bg_L, tot_L, rng)
    plot_spectrum_subplot(axes[i], bin_centers, bg_L, sig_L, tot_L, obs_bg, obs_tot, L)

if len(luminosities) < len(axes):
    axes[-1].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)

fig.suptitle("Diphoton Invariant Mass Spectra at Different Luminosities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("diphoton_spectra.png", dpi=300, bbox_inches='tight')
plt.show()

def plot_background_subtracted_subplot(ax, bin_centers, obs_tot, bg_L, sig_L, L):
    y = obs_tot - bg_L
    yerr = np.sqrt(obs_tot)

    ax.errorbar(bin_centers, y, yerr=yerr, fmt='o', capsize=3, label="Data − background")

    m_fine = np.linspace(m_min, m_max, 2000)
    smooth_sig = L * smooth_signal_curve(m_fine, m_h, sigma, A_signal)
    ax.plot(m_fine, smooth_sig, linewidth=2, label="Expected signal")

    ax.axhline(0, linewidth=1)

    ax.set_title(f"L = {L} fb$^{{-1}}$")
    ax.set_xlabel(r"$m_{\gamma\gamma}$ (GeV)")
    ax.set_ylabel(r"Counts $-$ background")
    ax.grid(alpha=0.3)

luminosities = [0.01, 1, 5, 25]
rng = np.random.default_rng(seed=123)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, L in enumerate(luminosities):
    bg_L, sig_L, tot_L = scale_with_luminosity(L, B_background, S_signal)
    obs_bg, obs_tot = poisson_sample(bg_L, tot_L, rng)
    plot_background_subtracted_subplot(axes[i], bin_centers, obs_tot, bg_L, sig_L, L)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)

fig.suptitle("Background-Subtracted Diphoton Mass Spectra", fontsize=16)

plt.tight_layout(rect=[0, 0.05, 1, 0.90])
fig.savefig("background_subtracted_panels.png", dpi=300, bbox_inches='tight')
plt.show()
