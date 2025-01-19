## This module contains a model predictive controller specific to a quadcopter system.
import numpy as np

# Define the controller.
class ModelPredictiveControl:
    
    # Intialisation function. Initialised the state matricies, etc.
    def __init__(self, A, B, C, f, v, W3, W4, x0, desired_control_trajectory_total):
        self.A = A # State transition matrix.
        self.B = B # Control Input matrix.
        self.C = C # Maps the systems state to its outout (Observables)
        self.f = f # Prediction Horizon
        self.v = v # Control horizon
        self.W3 = # Weight matrix, define how deviations from the desired trajectory are penalized
        self.W4 = W4 # Weight matrix
        self.desired_control_trajectory_total = desired_control_trajectory_total # Control objectives over time.
        self.n = A.shape[0]
        self.r = C.shape[0]
        self.m = B.shape[1]
        self.currentTimeStep = 0
        self.states = [x0]
        self.outputs = []
        self.inputs = []
        self.O, self.M, self.gainMatrix = self.formLiftedMatrices()

    # This function creates 'lifted' matricies O and M whicgh are used in the MPC optimisation. 
    def formLiftedMatrices(self):
        f = self.f
        v = self.v
        r = self.r
        n = self.n
        m = self.m
        A = self.A
        B = self.B
        C = self.C

        # Set O, this is the systems output for future steps based on the current state.
        O = np.zeros((f * r, n))
        
        # Loop through the prediction horizon.
        for i in range(f):
            
            # If the first element in the horizon is being processed, set the power of A as A.
            if i == 0:
                powA = A
            else:
                
                # Calculate the power of A (A^2, A^3...)
                powA = np.linalg.matrix_power(A, i)
            
            # Update the predicted output matrix    
            O[i * r : (i + 1) * r, :] = np.dot(C, powA)

        # Initialise the M matrix.
        M = np.zeros((f * r, v * m))
        
        # Loop through the prediction horsizon.
        for i in range(f):
            
            # Loop through the control horizon.
            for j in range(min(v, i + 1)):
                
                # If it is the 1st element, set as I.
                if j == 0:
                    powA = np.eye(n)
                else:
                    
                    # Calculate the matrix power for each iteration.
                    powA = np.linalg.matrix_power(A, j)
                    
                # Update the M matrix.
                M[i * r : (i + 1) * r, (i - j) * m :(i - j + 1) * m] = np.dot(C, np.dot(powA, B))

        # Compute the gaim matrix.
        tmp1 = np.dot(M.T, np.dot(self.W4, M))
        tmp2 = np.linalg.inv(tmp1 + self.W3)
        gainMatrix = np.dot(tmp2, np.dot(M.T, self.W4))

        return O, M, gainMatrix

    # This function simulates the next state of the system given the current state and control inpit.
    def propagateDynamics(self, controlInput, state):
        
        
        controlInput = controlInput[: self.B.shape[0]]  # Remove unnecessary indexing
        
        # Update the x state. (Ax + Bu)
        xkp1 = np.dot(self.A, state) + np.dot(self.B, controlInput)
        
        # Update the outut state. (Cx)
        yk = np.dot(self.C, state)
        return xkp1, yk
        
    # Compute the required control inputs.
    def computeControlInputs(self, estimated_state):
        
            # Get the desired control trajectroy for the current horizon.
            desiredControlTrajectory = self.desired_control_trajectory_total[self.currentTimeStep : self.currentTimeStep + self.f]
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