import numpy as np
import pandas as pd


# Convex function and gradient
def f2(x, y):
    return x**2 + y**2

def grad_f2(x, y):
    return 2*x, 2*y

# Momentum GD with per-iteration tracking
def momentum_gd_convex_table(start=(5.0, 5.0), lr=0.1, beta=0.9, n_iter=30):
    x, y = start
    vx, vy = 0.0, 0.0
    rows = []
    # record iteration 0 (initial state)
    gx, gy = grad_f2(x, y)
    rows.append([0, x, y, f2(x, y), gx, gy, vx, vy])
    
    for t in range(1, n_iter+1):
        gx, gy = grad_f2(x, y)
        vx = beta * vx - lr * gx
        vy = beta * vy - lr * gy
        x += vx
        y += vy
        fx = f2(x, y)
        gx_new, gy_new = grad_f2(x, y)
        rows.append([t, x, y, fx, gx_new, gy_new, vx, vy])
        
    cols = ["iter", "x", "y", "f(x,y)", "grad_x", "grad_y", "v_x", "v_y"]
    df = pd.DataFrame(rows, columns=cols)
    return df

df = momentum_gd_convex_table(start=(5.0, 5.0), lr=0.01, beta=0.8, n_iter=30)

# Round for readability
df_rounded = df.round(6)

# Save to CSV for download
csv_path = "momentum_convex_iterations.csv"
print(df_rounded)

