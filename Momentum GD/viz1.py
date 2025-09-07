import numpy as np
import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Define x values for the curve
x = np.linspace(0, 10, 1000)

# Create the saddle point landscape
# Left valley
left_valley = np.where(x <= 1.5, 
                      2 - 2*np.exp(-(x-0.5)**2/0.2), 
                      np.inf)

# Left slope up
left_slope = np.where((x > 1.5) & (x <= 2.5),
                     2 + 1.5*(x-1.5),
                     np.inf)

# Flat saddle region (plateau)
saddle_flat = np.where((x > 2.5) & (x <= 5.5),
                      5.25,
                      np.inf)

# Right slope down to valley
right_slope_down = np.where((x > 5.5) & (x <= 6.5),
                           5.25 + 1.5*(x-5.5),
                           np.inf)

# Right peak
right_peak = np.where((x > 6.5) & (x <= 7.5),
                     6.75 - 1.5*(x-6.5)**2/0.5,
                     np.inf)

# Right slope down
right_slope_final = np.where((x > 7.5) & (x <= 8.5),
                            6.75 - 2*(x-7.5),
                            np.inf)

# Final valley
final_valley = np.where(x > 8.5,
                       4.75 - 4*np.exp(-(x-9.5)**2/0.3),
                       np.inf)

# Combine all segments
y = np.minimum.reduce([left_valley, left_slope, saddle_flat, 
                       right_slope_down, right_peak, right_slope_final, final_valley])

# Remove infinite values and create clean curve
valid_mask = ~np.isinf(y)
x_clean = x[valid_mask]
y_clean = y[valid_mask]

# Create a smoother, more stylized version
x_smooth = np.array([0, 0.5, 1.5, 2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10])
y_smooth = np.array([8, 0.2, 3.75, 5.25, 5.25, 6.75, 4.75, 0.5, 8, 12])

# Interpolate for smoothness
from scipy.interpolate import interp1d
f = interp1d(x_smooth, y_smooth, kind='cubic')
x_final = np.linspace(0, 10, 500)
y_final = f(x_final)

# Plot the main curve
ax.plot(x_final, y_final, 'black', linewidth=3)

# Add the saddle point marker
saddle_x = 4  # Middle of flat region
saddle_y = 5.25
ax.plot(saddle_x, saddle_y, 'o', color='lightblue', markersize=12, 
        markeredgecolor='blue', markeredgewidth=2)

# Add the saddle point label with arrow
ax.annotate('Saddle Point', 
            xy=(saddle_x, saddle_y), 
            xytext=(saddle_x, saddle_y + 2),
            fontsize=14, 
            ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Style the plot to match your sketch
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1, 10)
ax.set_aspect('equal')

# Remove axes ticks and labels for clean look
ax.set_xticks([])
ax.set_yticks([])

# Add subtle grid or remove axes entirely for sketch-like appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Set background color to match sketch
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.title('Saddle Point Problem: Flat Plateau Traps Gradient Descent', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# Alternative version with gradient descent visualization
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

# Plot the same curve
ax2.plot(x_final, y_final, 'black', linewidth=3)

# Add saddle point
ax2.plot(saddle_x, saddle_y, 'o', color='lightblue', markersize=12, 
         markeredgecolor='blue', markeredgewidth=2)

# Simulate gradient descent getting stuck
# Start from a point that will get trapped
start_x = 3.8
start_y = f(start_x)
ax2.plot(start_x, start_y, 'ro', markersize=10, label='Starting point')

# Show very small movements (gradient descent steps)
steps_x = [3.8, 3.85, 3.9, 3.93, 3.95, 3.96, 3.97, 3.975, 3.98, 3.985]
for i, step_x in enumerate(steps_x):
    step_y = f(step_x)
    if i == 0:
        ax2.plot(step_x, step_y, 'ro', markersize=8)
    else:
        ax2.plot(step_x, step_y, 'r.', markersize=6, alpha=0.7)
    
    # Draw tiny arrows showing movement
    if i > 0:
        prev_x, prev_y = steps_x[i-1], f(steps_x[i-1])
        ax2.annotate('', xy=(step_x, step_y), xytext=(prev_x, prev_y),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.6, lw=1))

# Add annotations
ax2.annotate('Saddle Point\n(Zero Gradient)', 
            xy=(saddle_x, saddle_y), 
            xytext=(saddle_x + 1, saddle_y + 1.5),
            fontsize=12, 
            ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax2.annotate('Gradient Descent\ngets stuck here!\n(Vanishing gradients)', 
            xy=(steps_x[-1], f(steps_x[-1])), 
            xytext=(steps_x[-1] - 1.5, f(steps_x[-1]) + 1.5),
            fontsize=10, 
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Style the plot
ax2.set_xlim(-0.5, 10.5)
ax2.set_ylim(-1, 10)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_facecolor('white')

plt.title('Why Gradient Descent Fails at Saddle Points', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("🎯 SADDLE POINT PROBLEM EXPLAINED:")
print("="*40)
print("• Gradient ≈ 0 in the flat region")
print("• Optimizer thinks it found minimum")
print("• Actually stuck on a plateau/saddle")
print("• Needs momentum or noise to escape")
print("\n💡 This is why modern optimizers use:")
print("• Momentum (SGD with momentum)")
print("• Adaptive learning rates (Adam)")
print("• Random noise injection")
print("• Learning rate scheduling")