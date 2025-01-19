import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Given matrix P from MATLAB
P = np.array([[-0.8540, -0.2617],
              [-0.2617, -1.1433]])

# Generate points on the unit circle
theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)

# Apply the transformation P^(1/2) to get points on the ellipsoid
points = P @ np.vstack([x, y])

# Plot the ellipsoid
fig, ax = plt.subplots()
ax.plot(points[0, :], points[1, :], label='Ellipsoid')

# Plot the unit circle for reference
ax.plot(x, y, label='Unit Circle', linestyle='--')

# Set aspect ratio to be equal
ax.set_aspect('equal', adjustable='box')

# Set labels and legend
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()

plt.title('Ellipsoidal Invariant Set')
plt.grid(True)
plt.show()
