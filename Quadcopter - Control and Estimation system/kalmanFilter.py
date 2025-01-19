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

    def state_update(self, z):
        w = np.random.normal(0, np.sqrt(self.Q))
        z_next = self.A @ z + w * self.G
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
t_sim = 80
z_actual_cache = np.zeros((t_sim + 1, 2))
z_corr_cache = np.zeros((t_sim, 2))
z_pred_cache = np.zeros((t_sim + 1, 2))
sigma_pred_cache = np.zeros((2, 2, t_sim + 1))  # 3D array to hold covariance matrices over time

z_actual = np.random.multivariate_normal(initial_state.flatten(), P0).reshape(-1, 1)
z_actual_cache[0, :] = z_actual.T

for t in range(t_sim):
    # Output measurement
    y = kf.output(z_actual)

    # Measurement update
    z_corr, sigma_corr = kf.measurement_update(y, kf.z_pred, kf.sigma_pred)
    z_corr_cache[t, :] = z_corr.T

    # Time update
    z_pred, sigma_pred = kf.time_update(z_corr, sigma_corr)
    z_pred_cache[t + 1, :] = z_pred.T
    sigma_pred_cache[:, :, t + 1] = sigma_pred

    # State update
    z_actual = kf.state_update(z_actual)
    z_actual_cache[t + 1, :] = z_actual.T

# Plotting
time_scale = np.arange(t_sim + 1) * t_sampling
plt.plot(time_scale[:-1], z_corr_cache[:, 0], label="Corrected Position")
plt.plot(time_scale, z_actual_cache[:, 0], label="Actual Position", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid()
plt.legend()
plt.show()
