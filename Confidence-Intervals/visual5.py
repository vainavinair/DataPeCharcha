import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(23)

mu_true = 50
n = 80
alpha = 0.05
crit = norm.ppf(1 - alpha/2)

# Low vs High variability samples
sigma_low, sigma_high = 3, 12
x_low  = np.random.normal(mu_true, sigma_low,  n)
x_high = np.random.normal(mu_true, sigma_high, n)

# Stats
xbar_low,  s_low  = x_low.mean(),  x_low.std(ddof=1)
xbar_high, s_high = x_high.mean(), x_high.std(ddof=1)

me_low  = crit * s_low  / np.sqrt(n)
me_high = crit * s_high / np.sqrt(n)

fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

# Low variability panel
axs[0].scatter(x_low, np.zeros_like(x_low), s=12, alpha=0.7)
axs[0].hlines(0.2, xbar_low - me_low, xbar_low + me_low, linewidth=8, color="#88ccee")
axs[0].plot(xbar_low, 0.2, "o", color="#223344")
axs[0].set_title("Low Variability (Narrow CI)", fontsize=11, fontweight="bold")

# High variability panel
axs[1].scatter(x_high, np.zeros_like(x_high), s=12, alpha=0.7)
axs[1].hlines(0.2, xbar_high - me_high, xbar_high + me_high, linewidth=8, color="#88ccee")
axs[1].plot(xbar_high, 0.2, "o", color="#223344")
axs[1].set_title("High Variability (Wide CI)", fontsize=11, fontweight="bold")

# Cosmetics
for ax in axs:
    ax.set_xlabel("Value")
    ax.set_yticks([])
    for spine in ["top","left","right","bottom"]:
        ax.spines[spine].set_visible(False)

fig.suptitle("Same n, Same Confidence — Variability Alone Widens the CI", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
