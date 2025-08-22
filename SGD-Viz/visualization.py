import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define convex function: f(w1, w2) = (w1-2)^2 + (w2-3)**2
def f(w1, w2):
    return (w1 - 2)**2 + (w2 - 3)**2

# Gradient of the function
def grad(w):
    return 2 * (w - np.array([2, 3]))

# Simulate mini-batch noisy gradient
def noisy_grad(w, noise_scale=2.5):
    true_grad = grad(w)
    noise = np.random.normal(scale=noise_scale, size=w.shape)
    return true_grad + noise

# Parameters
lr = 0.05
steps_clean = 50   # GD takes more epochs
steps_noisy = 20   # SGD stops earlier
noise_scale = 2.5

# Starting point
start_w = np.array([6.0, 6.0])

# ----- Noisy SGD path (fewer epochs) -----
w = start_w.copy()
path_noisy = [w.copy()]
for _ in range(steps_noisy):
    g = noisy_grad(w, noise_scale)
    w = w - lr * g
    path_noisy.append(w.copy())
path_noisy = np.array(path_noisy)

# ----- Clean Gradient Descent path (more epochs) -----
w = start_w.copy()
path_clean = [w.copy()]
for _ in range(steps_clean):
    g = grad(w)  # no noise
    w = w - lr * g
    path_clean.append(w.copy())
path_clean = np.array(path_clean)

# Prepare meshgrid for surface plot
w1 = np.linspace(-1, 7, 100)
w2 = np.linspace(-1, 7, 100)
W1, W2 = np.meshgrid(w1, w2)
Z = f(W1, W2)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface
ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.6)

# Paths
ax.plot(path_noisy[:, 0], path_noisy[:, 1], f(path_noisy[:, 0], path_noisy[:, 1]),
        'ro-', label=f'Noisy SGD Path ({steps_noisy} epochs)')
ax.plot(path_clean[:, 0], path_clean[:, 1], f(path_clean[:, 0], path_clean[:, 1]),
        'bo-',alpha=0.5, label=f'Clean GD Path ({steps_clean} epochs)')

# Points
ax.scatter(path_noisy[0, 0], path_noisy[0, 1], f(path_noisy[0, 0], path_noisy[0, 1]),
           color='red', s=100, label='Start')
ax.scatter(2, 3, 0, color='green', s=100, label='Global Minimum')

# Labels
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.set_title('Noisy SGD (fewer epochs) vs Clean GD (more epochs)')
ax.legend()

plt.show()
