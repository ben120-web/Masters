import numpy as np
import matplotlib.pyplot as plt

# Define the missing initial conditions and parameters
x_0 = np.array([0.0, 0.0])  # Initial true state
P_0 = np.eye(2) * 0.1       # Initial estimate covariance
N = 100                    # Number of time steps

# Correct system parameters
A = np.array([[0.9, 0.2], [-0.1, 0.8]])
B = np.array([[0.1], [0.05]])
C = np.array([1.0, 0.0])
Q = np.array([[0.01, 0], [0, 0.01]])
R = 0.05
K = np.array([-2, -3])  # Stabilising gain

# Simulation parameters
x_est = x_0.copy()  # Copy to ensure x_0 is not modified
P_est = P_0.copy()
true_states = np.zeros((N, 2))  # True state
estimated_states = np.zeros((N, 2))  # Estimated state
measurements = np.zeros(N)  # Measurements
control_inputs = np.zeros(N)  # Control inputs

for t in range(N):
    # System dynamics
    if t == 0:
        true_states[t] = x_0
    else:
        true_states[t] = A.dot(true_states[t-1]) + B.dot([control_inputs[t-1]]) + np.random.multivariate_normal([0, 0], Q)
    
    # Measurements
    measurements[t] = C.dot(true_states[t]) + np.random.normal(0, np.sqrt(R))
    
    # Kalman Filter Prediction
    x_pred = A.dot(x_est) + B.dot([control_inputs[t-1]])
    P_pred = A.dot(P_est).dot(A.T) + Q
    
    # Kalman Filter Update
    S = C.dot(P_pred).dot(C.T) + R
    K_gain = P_pred.dot(C.T) / S
    x_est = x_pred + K_gain * (measurements[t] - C.dot(x_pred))
    P_est = P_pred - np.outer(K_gain, C).dot(P_pred)
    
    # Store estimated states
    estimated_states[t] = x_est
    
    # Determine control action based on estimated state
    control_inputs[t] = K.dot(x_est)

# Plotting
plt.figure(figsize=(18, 12))

# Measured outputs
plt.subplot(3, 1, 1)
plt.plot(measurements, label='Measured Output $y_t$')
plt.title('Measured Output $y_t$')
plt.legend()

# Estimated states
plt.subplot(3, 1, 2)
plt.plot(estimated_states[:, 0], label='$\hat{x}_{t|t}[0]$')
plt.plot(estimated_states[:, 1], label='$\hat{x}_{t|t}[1]$')
plt.title('State Estimates $\hat{x}_{t|t}$')
plt.legend()

# Control inputs
plt.subplot(3, 1, 3)
plt.plot(control_inputs, label='$u_t$')
plt.title('Control Inputs $u_t$')
plt.legend()

plt.tight_layout()
plt.show()
