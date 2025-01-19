import numpy as np
import matplotlib.pyplot as plt

# Define the system matrix A
A = np.array([[0.88, -0.2],
              [0.2, 0.88]])

# Set the initial state x0
x0 = np.array([1.5, -1.5])

# Set the number of time steps
N = 50

# Initialize arrays to store the coordinates of xt and x2,t
xt_coordinates = np.zeros((N+1, 2))
x2t_coordinates = np.zeros(N)

# Perform the simulation
xt_coordinates[0] = x0
for t in range(1, N+1):
    xt_coordinates[t] = np.dot(A, xt_coordinates[t-1])
    x2t_coordinates[t-1] = xt_coordinates[t, 1]

# Plot the coordinates of xt against time
plt.figure(figsize=(10, 6))
plt.plot(range(N+1), xt_coordinates[:, 0], label='x1,t')
plt.plot(range(N+1), xt_coordinates[:, 1], label='x2,t')
plt.title('Coordinates of xt against Time')
plt.xlabel('Time')
plt.ylabel('Coordinate Value')
plt.legend()
plt.show()

# Plot x2,t against x1,t
plt.figure(figsize=(8, 8))
plt.scatter(xt_coordinates[:, 0], x2t_coordinates, marker='o', label='x2,t against x1,t')
plt.title('Scatter Plot of x2,t against x1,t')
plt.xlabel('x1,t')
plt.ylabel('x2,t')
plt.legend()
plt.show()
