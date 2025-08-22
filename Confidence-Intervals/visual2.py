import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# True population value (unknown to us in reality)
true_value = 50  

# Number of samples to simulate
num_samples = 20

# Generate confidence intervals
ci_centers = np.random.normal(true_value, 2, size=num_samples)  # sample means
ci_half_widths = np.random.uniform(1.5, 3, size=num_samples)    # CI range half-widths

# Pick one interval to "miss" on purpose
miss_index = np.random.choice(num_samples, 1, replace=False)
ci_centers[miss_index] += 5  # shift it far enough to miss

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.axvline(true_value, color='red', linestyle='--', linewidth=2, label='True Population Value')

for i in range(num_samples):
    lower = ci_centers[i] - ci_half_widths[i]
    upper = ci_centers[i] + ci_half_widths[i]
    color = 'skyblue' if i != miss_index else 'orange'
    ax.hlines(y=i, xmin=lower, xmax=upper, color=color, linewidth=3)
    ax.plot(ci_centers[i], i, 'o', color='black')  # sample mean

# Labels & style
ax.set_yticks(range(num_samples))
ax.set_yticklabels([f"Sample {i+1}" for i in range(num_samples)])
ax.set_xlabel('Value')
ax.set_title("95% Confidence Intervals from Repeated Samples", fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()

# Caption
plt.figtext(0.5, -0.02, 
            "Our 95% confidence is in the method that produces these intervals, not in any single one.",
            ha="center", fontsize=10, color='gray')

plt.show()
