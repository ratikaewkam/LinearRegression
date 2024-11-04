import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Mean squared error
def MSE(x, y, m, b):
    total_error = 0
    n = len(x)
    for i in range(n):
        y_pred = (m*x[i]) + b
        total_error += np.power((y[i]-y_pred), 2)
    mse = total_error/n
    return mse

# Gradient descent
def gradient_descent(x, y, m_now, b_now, alpha):
    dm = 0
    db = 0
    n = len(x)
    for i in range(n):
        y_pred = (m_now*x[i]) + b_now
        dm += (-2/n) * (y[i]-y_pred) * x[i] # Derivative of mean squared error (m)
        db += (-2/n) * (y[i]-y_pred) # Derivative of mean squared error (b)
    m = m_now - alpha*dm
    b = b_now - alpha*db

    return m, b

# Read CSV file
df = pd.read_csv("data.csv")
x = df.iloc[0:132, 0].astype(int).to_numpy() # Col1
y = df.iloc[0:132, 1].astype(int).to_numpy() # Col2

m = -9.5
b = 0
zs = MSE(x, y, m, b)

m_vals = np.linspace(-20, 20)
b_vals = np.linspace(-20, 20)
M, B = np.meshgrid(m_vals, b_vals)
Z = np.array([np.mean((y - ((m*x) + b)) ** 2) for m, b in zip(M.ravel(), B.ravel())])
Z = Z.reshape(M.shape)

alpha = 0.001 # Learning rate

ax = plt.subplot(projection="3d", computed_zorder=False)


for i in range(17000):
    m, b = gradient_descent(x, y, m, b, alpha)
    zs = MSE(x, y, m, b)

    # Plot 3D
    ax.plot_surface(M, B, Z, cmap="plasma", zorder=0)
    ax.set_xlabel("m")
    ax.set_ylabel("b")
    ax.set_zlabel("MSE")
    ax.scatter(m, b, zs, color="red", zorder=1)
    plt.pause(0.001)
    ax.clear()
    print(m, b, zs)

# Plot 3D
ax.plot_surface(M, B, Z, cmap="plasma", zorder=0)
ax.set_xlabel("m")
ax.set_ylabel("b")
ax.set_zlabel("MSE")
ax.scatter(-9.5, 0, MSE(x, y, -9.5, 0), color="black", zorder=1)
ax.scatter(m, b, zs, color="red", zorder=1)
plt.show()

# Plot graph
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(list(range(1, 53)), [(m*x) + b for x in range(1, 53)], color="red") # Trendline
plt.show()