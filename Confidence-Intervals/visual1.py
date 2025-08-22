import matplotlib.pyplot as plt

# Data
sample_average = 32
ci_lower = 30
ci_upper = 34

# Create figure
fig, ax = plt.subplots(figsize=(8, 2.5))
fig.patch.set_facecolor('white')

# Number line
ax.hlines(y=0, xmin=28, xmax=36, color='#d9d9d9', linewidth=2, zorder=1)

# Confidence interval bar
ax.hlines(y=0, xmin=ci_lower, xmax=ci_upper, 
          color='#66b3ff', linewidth=8, alpha=0.8, zorder=2)

# Sample average point
ax.plot(sample_average, 0, 'o', color='#005c99', markersize=12, zorder=3)

# Labels
ax.text(sample_average, 0.35, 'Sample Average\n(Our Best Guess)', 
        ha='center', fontsize=10, color='#00334d', fontweight='bold')
ax.text((ci_lower + ci_upper)/2, -0.45, 'Confidence Interval\n(Likely Range)', 
        ha='center', fontsize=10, color='#333333')

# Aesthetic cleanup
ax.set_yticks([])
ax.set_xticks(range(28, 37))
ax.set_xlim(28, 36)
ax.set_ylim(-1, 1)

# Remove spines
for spine in ['top', 'left', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()
