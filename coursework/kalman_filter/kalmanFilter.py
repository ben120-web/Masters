import numpy as np
import matplotlib.pyplot as plt

# System parameters
A = 1  # System dynamics
C = 1  # Measurement matrix
Q = 0.1  # Process noise variance
R = 1  # Measurement noise variance
P_0 = 0.1  # Initial state variance
x_0 = 0  # True initial state
N = 100  # Number of time steps
n_sensors = [1, 3]  # Number of sensors for comparison

# Initialize state and measurement arrays
true_states = np.zeros(N)
measurements = np.zeros((N, max(n_sensors)))
estimated_states = np.zeros((N, len(n_sensors)))

# Process and measurement noise
process_noise = np.random.normal(0, np.sqrt(Q), N)
measurement_noise = np.random.normal(0, np.sqrt(R), (N, max(n_sensors)))

# Generate true states and measurements
true_states[0] = x_0
for t in range(1, N):
    true_states[t] = A * true_states[t-1] + process_noise[t-1]

for i in range(max(n_sensors)):
    measurements[:, i] = true_states + measurement_noise[:, i]

# Kalman filter implementation
def kalman_filter(n_sensor):
    estimated_state = np.zeros(N)
    P = P_0  # Initial estimate covariance
    for t in range(N):
        # Prediction step
        x_pred = A * estimated_state[t-1]
        P_pred = A * P * A + Q

        # Update step with n_sensor measurements
        y_avg = np.mean(measurements[t, :n_sensor])  # Average of n_sensor measurements
        K = P_pred * C / (C * P_pred * C + R / n_sensor)  # Kalman gain, adjusted for n_sensor
        estimated_state[t] = x_pred + K * (y_avg - C * x_pred)
        P = (1 - K * C) * P_pred

    return estimated_state

# Run Kalman filter for 1 and N sensors
for i, n_sensor in enumerate(n_sensors):
    estimated_states[:, i] = kalman_filter(n_sensor)

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(true_states, label='True State', color='k')
for i, n_sensor in enumerate(n_sensors):
    plt.plot(estimated_states[:, i], label=f'Estimate with {n_sensor} sensors')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('State Estimation with Different Number of Sensors')
plt.legend()
plt.show()
