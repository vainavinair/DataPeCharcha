import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(11)

# Fix a single sample to keep data constant across panels
mu_true = 50
sigma_true = 10
n = 200
x = np.random.normal(mu_true, sigma_true, n)
xbar, s = x.mean(), x.std(ddof=1)

# Two confidence levels (z-approx is fine for this n)
alpha_90, alpha_99 = 0.10, 0.01
crit_90 = norm.ppf(1 - alpha_90/2)
crit_99 = norm.ppf(1 - alpha_99/2)

me_90 = crit_90 * s / np.sqrt(n)
me_99 = crit_99 * s / np.sqrt(n)

fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

# 90%
axs[0].hlines(0, xbar - me_90, xbar + me_90, linewidth=8, color="#a1d99b")
axs[0].plot(xbar, 0, "o", color="#225522")
axs[0].set_title("90% Confidence (Narrower)", fontsize=11, fontweight="bold")

# 99%
axs[1].hlines(0, xbar - me_99, xbar + me_99, linewidth=8, color="#fc9272")
axs[1].plot(xbar, 0, "o", color="#7f0000")
axs[1].set_title("99% Confidence (Wider)", fontsize=11, fontweight="bold")

for ax in axs:
    ax.set_xlabel("Value")
    ax.set_yticks([])
    for spine in ["top","left","right","bottom"]:
        ax.spines[spine].set_visible(False)

fig.suptitle("Higher Confidence ⇒ Larger Critical Value ⇒ Wider CI", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
