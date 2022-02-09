import numpy as np
import sympy
import control
import SEA_dynamics

# Pull in old symbolic variables
q, q_dot, q_ddot = SEA_dynamics.q, SEA_dynamics.q_dot, SEA_dynamics.q_ddot

# Define constants
# We're defining numerical values for our constants because symbolically
# computing eigenvalues, defining controllers, etc is hard for the symbolic
# toolbox to handle with matrices of this size
constants = [1, 1, 1, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 9.81] 
tau = sympy.Matrix(sympy.symbols("tau(1:3)"))

# Extract the dynamics from our previous solutions
# Here I'm using Roy Featherstone's notation:
# 
#                             B*tau = D*ddq + H
# 
#  Where H = C*dq + N     (and N = G + damping)
#  And B is the input matrix, which is different from the B used in damping
#  The input matrix allows us to specify which degrees of freedom are
#  actuated, and which are not
vars = constants + [q, q_dot, q_ddot]
LHS = SEA_dynamics.LHS_Dampers_func(*vars)
D = LHS.jacobian(q_ddot)
H = LHS - D*q_ddot
B = sympy.Matrix([
    [1, 0],
    [0, 0],
    [0, 1],
    [0, 0]
])

# Forward Dynamics
# Here we define the forward dynamics of the system in terms of the full 
# state x = [q; q_dot]
x = sympy.Matrix.vstack(q, q_dot)
x_dot = sympy.Matrix.vstack(q_dot, q_ddot)

# Here you should define the forward dynamics of your system x_dot = F(x, u)
# #### YOUR WORK FOR (a) HERE ####
C1 = sympy.Matrix([
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
    ])
C2 = sympy.Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ])
ForwardDynamics = C1 * x + C2 * D.inv() * B * tau - C2 * D.inv() * H
ForwardDynamics_func = sympy.lambdify([x, tau], ForwardDynamics, "numpy") # DD NOT MODIFY
#x_dot = ForwardDynamics

# Linearize Dynamics and Check Eigenvalues
# Now linearize about the equilibrium point (x, tau) = (0, 0)
# Lecture 4 and 5 slides both contain the formula
# The SymPy "jacobian" function will be very useful
# After this you should check the eigenvalues of your A matrix. How many of them have positive real 
# components?
# #### YOUR WORK FOR (b) HERE ####
Alin_sym = ForwardDynamics.jacobian(x) # symbolic expression for
Blin_sym = ForwardDynamics.jacobian(tau) # linearized system matrices
Alin_func = sympy.lambdify([x, tau], Alin_sym, "sympy")
Blin_func = sympy.lambdify([x, tau], Blin_sym, "sympy")
eq = [np.zeros(8), np.zeros(2)] # substitute in our equilibrium point
Alin = np.array(Alin_func(*eq)).astype(np.float64) # compute result as numpy array
Blin = np.array(Blin_func(*eq)).astype(np.float64) # use these for eigenvalue calculations
eigenvalues = np.linalg.eig(Alin)
eigenvalues = np.sort(eigenvalues) # For consistent autograder results, we sort these
print("Linearized System Eigenvalues:")
print(eigenvalues,"\n")

# Controller design
# Now that you've proved that the linearization is unstable (and therefore
# that the nonlinear dynamics are unstable), you can design a controller to
# stabilize these linear dynamics. Design an LQR controller using the
# following gains:
Q = np.eye(x.shape[0])
R = np.eye(tau.shape[0])
# What are the closed loop eigenvalues of your system? Is the controlled
# system stable?
#### YOUR WORK FOR (c) HERE ####
K = ...
closed_loop_eigenvalues = ...
closed_loop_eigenvalues = np.sort(closed_loop_eigenvalues) # sort again
print("Eigenvalues of Linearized System in Feedback:")
print(closed_loop_eigenvalues, "\n")

# Stability Bound
# What is the lower (slowest) bound of your controller's rate of
# convergence close to the origin?
#### YOUR WORK FOR (d) HERE ####
lower_bound = ...