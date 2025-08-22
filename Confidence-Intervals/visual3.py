import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

np.random.seed(7)

# True population parameters (unknown in reality)
mu_true = 50
sigma_true = 10

# Two sample sizes
n_small, n_large = 30, 1000
alpha = 0.05  # 95% CI
use_t_for_small = True

# Draw samples
x_small = np.random.normal(mu_true, sigma_true, n_small)
x_large = np.random.normal(mu_true, sigma_true, n_large)

# Stats
xbar_small, s_small = x_small.mean(), x_small.std(ddof=1)
xbar_large, s_large = x_large.mean(), x_large.std(ddof=1)

# Critical values
crit_small = t.ppf(1 - alpha/2, df=n_small-1) if use_t_for_small else norm.ppf(1 - alpha/2)
crit_large = norm.ppf(1 - alpha/2)

# Margins
me_small = crit_small * s_small / np.sqrt(n_small)
me_large = crit_large * s_large / np.sqrt(n_large)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, label, xbar, me in [
    (axs[0], f"Small Sample (N={n_small})", xbar_small, me_small),
    (axs[1], f"Large Sample (N={n_large})", xbar_large, me_large),
]:
    ax.hlines(0, xbar - me, xbar + me, linewidth=8, color="#88ccee")
    ax.plot(xbar, 0, "o", color="#223344")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_yticks([])
    for spine in ["top","left","right","bottom"]:
        ax.spines[spine].set_visible(False)

fig.suptitle("Sample Size Changes the Width (95% CI)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
