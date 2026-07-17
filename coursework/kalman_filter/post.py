import numpy as np
import matplotlib.pyplot as plt

# Define system matrices
A = np.array([[0.8, 0.2],
              [0.3, 0.5]])
B = np.array([[1],
              [0]])
C = np.array([[1, 0]])

# Define noise covariances
Q = np.array([[0.1, 0],
              [0, 0.1]])
R = 1

# Define stabilizing gain
K = np.array([[1, 1]])

# Initial state and covariance
x0 = np.array([[0],
               [0]])
P0 = np.eye(2) * 0.1  # Initial covariance

# Simulation parameters
T = 100  # Number of time steps

# Initialize arrays to store simulation results
x_true = np.zeros((2, T))
y_true = np.zeros(T)
x_est = np.zeros((2, T))
P_est = np.zeros((2, 2, T))
u = np.zeros(T)

# Initialize state and covariance estimates
x_est[:, 0] = x0[:, 0]
P_est[:, :, 0] = P0

# Simulate the closed-loop system
for t in range(T):
    # Generate process and measurement noise
    w = np.random.multivariate_normal([0, 0], Q)
    v = np.random.normal(0, R)

    # Simulate true system dynamics
    x_true[:, t+1] = A.dot(x_true[:, t]) + B.dot(u[t][0]) + w
    y_true[t] = C.dot(x_true[:, t]) + v

    # Prediction step of Kalman filter
    x_pred = A.dot(x_est[:, t]) + B.dot(u[t])
    P_pred = A.dot(P_est[:, :, t]).dot(A.T) + Q

    # Update step of Kalman filter
    Kt = P_pred.dot(C.T) / (C.dot(P_pred).dot(C.T) + R)
    x_est[:, t+1] = x_pred + Kt.dot(y_true[t] - C.dot(x_pred))
    P_est[:, :, t+1] = (np.eye(2) - Kt.dot(C)).dot(P_pred)

    # Compute control input
    u[t] = -K.dot(x_est[:, t+1])

# Plot results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(range(T), y_true, label='True Output')
plt.plot(range(T), C.dot(x_est), label='Estimated Output')
plt.title('Output ($y_t$) vs Time')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(T), x_true[0, :], label='True State 1')
plt.plot(range(T), x_est[0, :], label='Estimated State 1')
plt.plot(range(T), x_true[1, :], label='True State 2')
plt.plot(range(T), x_est[1, :], label='Estimated State 2')
plt.title('State ($x_t$) vs Time')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(T), u, label='Control Input')
plt.title('Control Input ($u_t$) vs Time')
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.legend()

plt.tight_layout()
plt.show()
