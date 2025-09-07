import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a U-shaped 3D valley: z = x^2 + 2*y^2
def f(x, y):
    return x**2 + 2*y**2

# Gradients
def grad(x, y):
    return np.array([2*x, 4*y])

# Vanilla Gradient Descent in 3D
def gd_3d(start, lr=0.1, steps=15):
    x, y = start
    traj = [np.array([x, y])]
    for _ in range(steps):
        g = grad(x, y)
        x -= lr * g[0]
        y -= lr * g[1]
        traj.append(np.array([x, y]))
    traj = np.array(traj)
    z = f(traj[:,0], traj[:,1])
    return traj, z

# Momentum Gradient Descent in 3D
def momentum_3d(start, lr=0.1, momentum=0.8, steps=15):
    x, y = start
    v = np.array([0.0, 0.0])
    traj = [np.array([x, y])]
    for _ in range(steps):
        g = grad(x, y)
        v = momentum * v - lr * g
        x += v[0]
        y += v[1]
        traj.append(np.array([x, y]))
    traj = np.array(traj)
    z = f(traj[:,0], traj[:,1])
    return traj, z

# Starting point
start = [3, 2]
traj_gd, z_gd = gd_3d(start)
traj_mom, z_mom = momentum_3d(start)

# Generate mesh for surface
x_vals = np.linspace(-4, 4, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# 3D plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Surface
ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', edgecolor='none')

# Trajectories
ax.plot(traj_gd[:,0], traj_gd[:,1], z_gd, 'ro-', lw=2, markersize=5, label='Vanilla GD')
ax.plot(traj_mom[:,0], traj_mom[:,1], z_mom, 'bo-', lw=2, markersize=5, label='Momentum')

# Start and minimum
ax.scatter(*start, f(*start), color='black', s=80, marker='X', label='Start')
ax.scatter(0,0,0, color='green', s=100, marker='*', label='Minimum')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Momentum vs Gradient Descent in a 3D U-Shaped Valley")
ax.legend()
plt.show()
