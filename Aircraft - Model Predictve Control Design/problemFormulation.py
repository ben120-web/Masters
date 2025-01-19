import qpsolvers as qp
from qpsolvers import solve_qp
import numpy as np

N = 10  # Prediction horizon
R = np.array([[1]])  # Control input weight
Q = np.diag([1, 1, 1])  # State deviation weight

# Construct the H matrix
H = np.kron(np.eye(N), R)

F = np.zeros(N)

# Assuming delta_min and delta_max are the bounds for the control input (elevator deflection)
delta_min = -24 * np.pi / 180  # converting degrees to radians
delta_max = 27 * np.pi / 180  # converting degrees to radians

# Control input constraints for each time step
G = np.vstack((np.eye(N), -np.eye(N)))
h = np.hstack((np.full(N, delta_max), np.full(N, -delta_min)))

# Solve the optimisation problem.
U_opt = solve_qp(P=H, q=F, G=G, h=h, solver='ecos')

print(U_opt)