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
        w = np.random.normal(0, np.sqrt(self.Q), size=(self.G.shape[1], 1))
        control_effect = np.array([[0], [control_input]])
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

class ModelPredictiveControl:
    def __init__(self, A, B, C, f, v, W3, W4, x0, desired_control_trajectory_total):
        self.A = A
        self.B = B
        self.C = C
        self.f = f
        self.v = v
        self.W3 = W3
        self.W4 = W4
        self.desired_control_trajectory_total = desired_control_trajectory_total
        self.n = A.shape[0]
        self.r = C.shape[0]
        self.m = B.shape[1]
        self.currentTimeStep = 0
        self.states = [x0]
        self.outputs = []
        self.inputs = []
        self.O, self.M, self.gainMatrix = self.formLiftedMatrices()

    def formLiftedMatrices(self):
        f = self.f
        v = self.v
        r = self.r
        n = self.n
        m = self.m
        A = self.A
        B = self.B
        C = self.C

        O = np.zeros((f * r, n))
        for i in range(f):
            if i == 0:
                powA = A
            else:
                powA = np.linalg.matrix_power(A, i)
            O[i*r:(i+1)*r, :] = np.dot(C, powA)

        M = np.zeros((f * r, v * m))
        for i in range(f):
            for j in range(min(v, i + 1)):
                if j == 0:
                    powA = np.eye(n)
                else:
                    powA = np.linalg.matrix_power(A, j)
                M[i*r:(i+1)*r, (i-j)*m:(i-j+1)*m] = np.dot(C, np.dot(powA, B))

        tmp1 = np.dot(M.T, np.dot(self.W4, M))
        tmp2 = np.linalg.inv(tmp1 + self.W3)
        gainMatrix = np.dot(tmp2, np.dot(M.T, self.W4))

        return O, M, gainMatrix

    def propagateDynamics(self, controlInput, state):
        controlInput = controlInput[:self.B.shape[0]]  # Remove unnecessary indexing
        xkp1 = np.dot(self.A, state) + np.dot(self.B, controlInput)
        yk = np.dot(self.C, state)
        return xkp1, yk
        
    def computeControlInputs(self, estimated_state):
            desiredControlTrajectory = self.desired_control_trajectory_total[self.currentTimeStep:self.currentTimeStep+self.f]
            desiredControlTrajectory = np.vstack((desiredControlTrajectory, np.zeros((self.f - len(desiredControlTrajectory), 1))))

            vectorS = desiredControlTrajectory - np.dot(self.O, estimated_state)

            # Calculate the deviation between estimated altitude and desired altitude
            altitude_deviation = estimated_state[0] - desiredControlTrajectory[0]

            # Adjust control inputs based on altitude deviation
            if abs(altitude_deviation) > 2:
                # If the deviation is greater than 2 meters, adjust the control input
                sign = np.sign(altitude_deviation)
                inputSequenceComputed = np.dot(self.gainMatrix, vectorS) - sign * 2  # Adjust control input
            else:
                inputSequenceComputed = np.dot(self.gainMatrix, vectorS)
            
            inputApplied = inputSequenceComputed[0]
            
            # Propagate dynamics using the computed control input
            state_kp1, output_k = self.propagateDynamics(inputApplied, self.states[self.currentTimeStep])
            
            # Update states, outputs, and inputs
            self.states.append(state_kp1)
            self.outputs.append(output_k)
            self.inputs.append(inputApplied)
            self.currentTimeStep += 1

# Define system matrices and initial conditions
t_sampling = 0.1
A = np.array([[1, t_sampling], [0, 1]])
B = np.array([[0.5 * t_sampling**2], [t_sampling]])
C = np.array([[1, 0]])
P0 = np.array([[100, 0], [0, 10]])
Q = 10  # Variance of process noise
R = 0.01  # Variance of measurement noise
initial_state = np.array([[0], [0]])

# Simulation parameters
desired_altitude = 40  # meters
f = 50  # Prediction horizon
v = 3  # Control horizon
W3 = np.eye(3) * 0.1  # Initialize W3 as a 3x3 identity matrix with a scaling factor
desired_control_trajectory_total = np.random.randn(f, 1)
W4 = np.eye(f * C.shape[0]) * 0.1  # Weight matrix for predicted output

# Initialize Kalman filter and MPC
kf = KalmanFilter(A, B, C, Q, R, P0, initial_state)
mpc = ModelPredictiveControl(A, B, C, f, v, W3, W4, initial_state, desired_control_trajectory_total)

# Simulation parameters
t_sim = 80
dt = 0.1

# Initialize variables for storing simulation data
time_stamps = np.arange(0, t_sim, dt)
altitude_estimates = []
throttle_adjustments = []

z_actual = np.random.multivariate_normal(initial_state.flatten(), P0).reshape(-1, 1)

for t in time_stamps:
    # Update Kalman filter
    z_actual = kf.state_update(z_actual, 0)  # No control input for now

    # Use estimated state for MPC
    mpc.computeControlInputs(z_actual)

    # Store the results for plotting
    altitude_estimates.append(z_actual[0])
    throttle_adjustments.append(mpc.inputs[-1][0])

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
