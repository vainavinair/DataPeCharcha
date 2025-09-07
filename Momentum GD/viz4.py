import numpy as np
import matplotlib.pyplot as plt

# Function: simple convex quadratic
def f(x):
    return x[0]**2 + x[1]**2

# Gradient of f
def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

def momentum_gradient_descent(beta, lr, iterations=30, start=np.array([4.0, 4.0])):
    x = start.copy()
    v = np.zeros_like(x)
    path = [x.copy()]

    for _ in range(iterations):
        g = grad_f(x)
        v = beta * v - lr * g  # momentum update
        x = x + v
        path.append(x.copy())

    return np.array(path)

# Parameters to test
betas = [0.0, 0.5, 0.9, 0.99]
lr = 0.01  # fixed learning rate

# Plot contours of the function
x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f([X1, X2])

plt.figure(figsize=(12, 10))
plt.contour(X1, X2, Z, levels=30, cmap='viridis')

# Plot paths for different momentum values
for beta in betas:
    path = momentum_gradient_descent(beta=beta, lr=lr, iterations=40)
    plt.plot(path[:,0], path[:,1], marker='o', label=f"β={beta}")

plt.scatter(0, 0, color='red', marker='*', s=200, label="Minimum")
plt.title("Momentum Gradient Descent Paths")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()
