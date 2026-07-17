import numpy as np
import matplotlib.pyplot as plt
 
class KalmanFilter:
    def __init__(self, A, G, C, Q, R, P0, initial_state):
        self.A = A
        self.G = G
        self.C = C
        self.Q = Q
        self.R = R
        self.sigma_pred = P0
        self.z_pred = initial_state

    def state_update(self, z, control_input):
        # Ensure w is a vector with the same number of rows as G has columns
        w = np.random.normal(0, np.sqrt(self.Q), size=(self.G.shape[1], 1))
        control_effect = np.array([[0], [control_input]])  # Assuming control affects the velocity
        z_next = self.A @ z + self.G @ w + control_effect
        return z_next.reshape(-1, 1)


    def output(self, z):
        v = np.random.normal(0, np.sqrt(self.R))
        y = self.C @ z + v
        return y

    def measurement_update(self, y, z_pred, sigma_pred):
        F = self.C @ sigma_pred @ self.C.T + self.R
        output_error = y - self.C @ z_pred
        K = sigma_pred @ self.C.T @ np.linalg.inv(F)
        z_corrected = z_pred + K @ output_error
        sigma_corrected = sigma_pred - K @ self.C @ sigma_pred
        return z_corrected, sigma_corrected

    def time_update(self, z_meas_update, sigma_meas_update):
        z_predicted = self.A @ z_meas_update
        sigma_predicted = self.A @ sigma_meas_update @ self.A.T + self.Q * (self.G @ self.G.T)
        return z_predicted, sigma_predicted

## This integrates the Kalman filter estimates and a PID controller.
class PIDController:
    def __init__(self, kp, ki, kd, set_point):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.integral = 0
        self.prev_error = 0

    def update(self, measured_value, dt):
        error = self.set_point - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
    
# Define system matrices and initial conditions
t_sampling = 0.1
A = np.array([[1, t_sampling], [0, 1]])
G = np.array([[0.5 * t_sampling**2], [t_sampling]])
C = np.array([[1, 0]])
P0 = np.array([[100, 0], [0, 10]])
Q = 10 # variance of process noise
R = 0.01 # variance of measurement noise
initial_state = np.array([[0], [0]])

kf = KalmanFilter(A, G, C, Q, R, P0, initial_state)

# Simulation parameters
desired_altitude = 40  # meters
kp, ki, kd = 5, 0, 0  # PID coefficients
pid = PIDController(kp, ki, kd, desired_altitude)

# Simulation parameters
t_sim = 80
z_actual_cache = np.zeros((t_sim + 1, 2))
z_corr_cache = np.zeros((t_sim, 2))
z_pred_cache = np.zeros((t_sim + 1, 2))
sigma_pred_cache = np.zeros((2, 2, t_sim + 1))  # 3D array to hold covariance matrices over time

z_actual = np.random.multivariate_normal(initial_state.flatten(), P0).reshape(-1, 1)
z_actual_cache[0, :] = z_actual.T

t_sim = 100  # total simulation time in seconds
dt = 0.1  # time step

# Initialize the Kalman Filter and PID Controller with the provided parameters
kf = KalmanFilter(A, G, C, Q, R, P0, initial_state)
pid = PIDController(kp, ki, kd, desired_altitude)

# Initialize variables for storing simulation data
time_stamps = np.arange(0, t_sim, dt)
altitude_estimates = []
throttle_adjustments = []

z_actual = np.random.multivariate_normal(initial_state.flatten(), P0).reshape(-1, 1)

for t in time_stamps:
    
    # Update Kalman Filter
    y = kf.output(z_actual)
    z_corr, sigma_corr = kf.measurement_update(y, kf.z_pred, kf.sigma_pred)
    kf.z_pred, kf.sigma_pred = kf.time_update(z_corr, sigma_corr)
    
    # PID Controller Update
    altitude_estimate = z_corr[0, 0]
    throttle_adjustment = pid.update(altitude_estimate, dt)
    
    # Simulate the altitude measurement
    z_actual = kf.state_update(z_actual, throttle_adjustment)
    
    # Store the results for plotting
    altitude_estimates.append(altitude_estimate)
    throttle_adjustments.append(throttle_adjustment)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_stamps, altitude_estimates, label='Estimated Altitude')
plt.plot(time_stamps, [desired_altitude] * len(time_stamps), 'r--', label='Desired Altitude')
plt.title('Altitude Control Simulation')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_stamps, throttle_adjustments, label='Throttle Adjustment')
plt.title('Throttle Adjustment Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Throttle Adjustment')
plt.legend()

plt.tight_layout()
plt.show()
    