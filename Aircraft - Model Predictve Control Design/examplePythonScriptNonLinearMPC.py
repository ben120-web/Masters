import numpy as np
import cvxpy as cp
import scipy as sp
import control as ctrl
import polytope as pc
import matplotlib.pyplot as plt
import scipy.linalg as spla

# Define the problem data
# -----------------------------
n = 2       # 2 states
m = 1       # 1 input
A = np.array([[1, 0.7], [-0.1, 1]])
B = np.array([[1], [0.5]])
N = 10

# Given matrices Q and R, solve DARE and determine
# a stabilising matrix K
# -----------------------------
Q = (2 ** 0.5)*np.eye(2)
R = 2 ** 0.5
P, _, K = ctrl.dare(A, B, Q, R)
K = -K
A_bar = A + B @ K
Q_bar = Q + K.T @ K

A_bar_1 = A_bar[0, 0]
A_bar_2 = A_bar[0, 1]
A_bar_3 = A_bar[1, 0]
A_bar_4 = A_bar[1, 1]
# let (A_bar.T @ P @ A_bar - P) as a_cal and (-2Q_bar) as b_cal to build a linear equation to solve
a_cal = np.array([[A_bar_1**2-1, A_bar_1*A_bar_3, A_bar_1*A_bar_3, A_bar_3**2], [A_bar_1*A_bar_2, A_bar_1*A_bar_4-1, A_bar_2*A_bar_3, A_bar_3*A_bar_4], [A_bar_1*A_bar_2, A_bar_2*A_bar_3, A_bar_1*A_bar_4-1, A_bar_3*A_bar_4], [A_bar_2**2, A_bar_2*A_bar_4, A_bar_2*A_bar_4, A_bar_4**2-1]])
b_cal = np.reshape(-2*Q_bar, (-1, 1))
P = sp.linalg.solve(a_cal, b_cal)
P = np.reshape(P, (2, -1))  # P must be symmetric!
# print("P=\n", P)
# P_check = A_bar.T @ P @ A_bar + 2*Q_bar  # check the equation P = A_bar.T @ P @ A_bar + 2*Q_bar
# print("check P:\n", P_check)

# Define the sets of constraints; it is
# X = {x : H_x * x <= b_x}
# U = {u : H_u * u <= b_u}
# -----------------------------
x_min = np.array([[-2], [-2]])
x_max = np.array([[2], [2]])
u_min = np.array([-1])
u_max = np.array([1])

# Problem statement
# -----------------------------
x0 = cp.Parameter(n)        # <--- x is a parameter of the optimisation problem P_N(x)
u_seq = cp.Variable((m, N))     # <--- sequence of control actions
x_seq = cp.Variable((n, N+1))

cost = 0
constraints = [x_seq[:, 0] == x0]       # x_0 = x
x_min = np.array([-2, -2])
x_max = np.array([2, 2])
u_min = np.array([-1])
u_max = np.array([1])
for t in range(N-1):
    xt_var = x_seq[:, t]      # x_t
    ut_var = u_seq[:, t]      # u_t
    cost += cp.norm2(xt_var)**2 + ut_var**2

    # dynamics, x_min <= xt <= x_max, u_min <= ut <= u_max
    constraints += [x_seq[:, t+1] == A@xt_var + B@ut_var,
                    x_min <= xt_var,
                    xt_var <= x_max,
                    u_min <= ut_var,
                    ut_var <= u_max]
# cost += 0     # the terminal cost V_f(x) = 0

xN = x_seq[:, N-1]
cost += 0.5*cp.quad_form(xN, P)     # terminal cost

constraints += [x_min <= xN, xN <= x_max]       # terminal constraints (x_min <= xN <= x_max)
# Compute alpha using Equation
H = np.eye(2)
H = np.vstack((H, K))
H = np.vstack((H, -np.eye(2)))
H = np.vstack((H, -K))  # H = [[I], [K], [-I], [-K]]
b = np.array([[1], [1], [1], [1], [1], [1]])    # b = [[x_{max}], [u_{max}], [-x_{min}], [-u_{min}]]

# figure out P_N^(-1/2) Note:P_N is a matrix!
# v is the eigenvalue, Q is the eigenvector
v, Q = np.linalg.eig(P)
# print(v)
# V is the diagonal matrix of v
V = np.diag(v**(-0.5))
# print(V)
# P_N = Q * V * Q^(-1)
P_alpha = np.dot(np.dot(Q, V, np.linalg.inv(Q)), np.linalg.inv(Q))

# calculate the minimum of alpha
for i in range(6):
    alpha = 1/(np.linalg.norm(P_alpha @ np.reshape(H[i], (2, 1)))**2)
    if i == 0:
        alpha_temp = alpha
    if alpha < alpha_temp:
        alpha_temp = alpha
    # print(alpha)
alpha = alpha_temp

# constraints of MPC
constraints_mpc = constraints + [cp.quad_form(xN, P) <= alpha]
problem = cp.Problem(cp.Minimize(cost), constraints_mpc)


def mpc(state):
    x0.value = state
    out = problem.solve()
    return u_seq[:, 0].value


# Solve the problem with MPC
# -----------------------------
# x_init = pc.extreme(X_i)[2]  # the extreme points of X_N
x_init = np.array([0.01, 0.04])     # any feasible initial states
x_current = x_init

N_sim = 40
u_cache = []    # a list to save u_mpc
x_cache = x_current     # a list to save x_t
V_N_cache = []  # a list to save the cost value V_N^star
for t in range(N_sim):
    u_mpc = mpc(x_current)
    u_cache.append(u_mpc)
    x_current = A @ x_current + B @ u_mpc
    x_cache = np.concatenate((x_cache, x_current))
    V_N_cache.append(cost.value)
x_cache = np.reshape(x_cache, (N_sim+1, n))

# Plotting of solution
# -----------------------------

plt.rcParams['font.size'] = '14'
plt.figure(1),
plt.title('States vs time')
plt.plot(x_cache[:, 0], label='x1')
plt.plot(x_cache[:, 1], label='x2')
plt.xlabel('Time, t')
plt.ylabel('States, x_t')
plt.legend()        # show labels

plt.figure(2)
plt.title('control actions vs time')
plt.plot(u_cache)
plt.xlabel('Time, t')
plt.ylabel('control actions, u_t')

plt.figure(3)
plt.title('V_N_star vs time')
plt.plot(V_N_cache)
plt.xlabel('Time, t')
plt.ylabel('V_N_star')

plt.figure(4)
plt.title('States situation')
plt.xlim([-0.1, 0.1])
plt.ylim([-0.1, 0.1])
plt.plot(x_cache[:, 0], x_cache[:, 1], '-o')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()