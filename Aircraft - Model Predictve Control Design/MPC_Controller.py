import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
import polytope as pc

# System matrices
A = np.array([[0.9835, 2.782, 0], [-0.0006821, 0.978, 0], [-0.0009730, 2.804, 1]])
B = np.array([[0.01293], [0.00100], [0.001425]])

# MPC parameters
N = 10  # Prediction horizon

# Constraints
delta_min, delta_max = -24 * np.pi / 180, 27 * np.pi / 180  # Elevator deflection angle bounds in radians
alpha_max = 11.5 * np.pi / 180  # Angle of attack bounds in radians
theta_max = 35 * np.pi / 180  # Pitch angle bounds in radians
q_max = 14 * np.pi / 180  # Pitch rate bounds in radians

# Initial state
x0 = np.array([1.0, 0.5, -0.5])

# Cost function weights
Q = np.diag([1000, 0, 0])  # State penalties
R = np.array([[1]])  # Control effort penalty

# Helper Functions
def build_prediction_matrices(A, B, N):
    n, m = A.shape[0], B.shape[1]
    Px = np.zeros((N*n, n))
    Pu = np.zeros((N*n, N*m))
    for i in range(N):
        Px[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            Pu[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B
    return Px, Pu

def build_cost_matrices(Q, R, N, m, Px, Pu, x_ref):
    Q_extended = np.kron(np.eye(N), Q)
    R_extended = np.kron(np.eye(N), R)
    H = Pu.T @ Q_extended @ Pu + R_extended
    F = np.zeros(N * m)  # Assuming no linear term; adjust as needed
    return H, F

def build_constraints_matrices(N, m, delta_min, delta_max, alpha_max, q_max, theta_max):
    # Control input constraints
    G_delta = np.vstack([np.eye(N*m), -np.eye(N*m)])
    h_delta = np.hstack([np.full(N*m, delta_max), np.full(N*m, -delta_min)])
    
    # State constraints (angle of attack, pitch rate, and pitch angle)
    # Note: Simplified for demonstration. In practice, state constraints need to be integrated based on system dynamics.
    G_state = np.zeros((0, N*m))  # Placeholder for state constraints
    h_state = np.array([])  # Placeholder for state constraint values
    
    G = np.vstack([G_delta, G_state])
    h = np.hstack([h_delta, h_state])
    
    return G, h

def compute_maximal_invariant_set(A, B, state_constraints, input_constraints, tol=1e-4, max_iter=100):
    """
    Compute the maximal invariant set for a given linear system and constraints.
    
    A, B: System matrices defining the dynamics x' = Ax + Bu.
    state_constraints: A Polytope object defining the state constraints.
    input_constraints: A Polytope object defining the input constraints.
    tol: Tolerance for checking set containment.
    max_iter: Maximum number of iterations for the algorithm.
    """
    # Initial set is the state constraint set
    Omega = state_constraints
    for _ in range(max_iter):
        pre_Omega = pc.pre(Omega, A, B, U=input_constraints, W=None)
        new_Omega = pc.intersect(Omega, pre_Omega)
        if pc.is_empty(new_Omega - Omega, tol):
            break
        Omega = new_Omega
    return Omega

def simulate_step(A, B, x, u):
    return A @ x + B @ u

def plot_results(x_history, u_history):
    t = np.arange(len(x_history))
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, x_history[:, 0], label='Angle of Attack')
    plt.plot(t, x_history[:, 1], label='Pitch Rate')
    plt.plot(t, x_history[:, 2], label='Pitch Angle')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.title('State Trajectories')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.step(t[:-1], u_history, where='post', label='Control Input')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input')
    plt.title('Control Input (Elevator Deflection Angle)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def run_simulation():
    n, m = A.shape[0], B.shape[1]
    Px, Pu = build_prediction_matrices(A, B, N)
    G, h = build_constraints_matrices(N, m, delta_min, delta_max, alpha_max, q_max, theta_max)
    x_history, u_history = [x0], []
    
    x = x0
    for _ in range(50):  # Simulation time steps
        H, F = build_cost_matrices(Q, R, N, m, Px, Pu, x)
        U_opt = solve_qp(H, F, G, h, solver='ecos')
        u = U_opt[0]  # First control input
        u_history.append(u)
        x = simulate_step(A, B, x, np.array([u]))
        x_history.append(x)
    
    return np.array(x_history), np.array(u_history)

x_history, u_history = run_simulation()
plot_results(x_history, u_history)
