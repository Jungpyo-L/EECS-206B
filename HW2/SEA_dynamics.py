import sympy
from sympy.core.function import diff

# Constants
l1 = sympy.Symbol("l1") # length of each arm
l2 = sympy.Symbol("l2")
m1 = sympy.Symbol("m1") # mass of each arm
m2 = sympy.Symbol("m2") 
I1 = sympy.Symbol("I1") # moment of inertia of each arm
I2 = sympy.Symbol("I2") 
J1 = sympy.Symbol("J1") # inertia of each motor
J2 = sympy.Symbol("J2") 
K1 = sympy.Symbol("K1") # spring constants
K2 = sympy.Symbol("K2") 
B1 = sympy.Symbol("B1")
B2 = sympy.Symbol("B2") 
g = sympy.Symbol("g") # acceleration due to gravity

# Define generalized coordinates as functions of time
# and specify their derivatives for prettier computation
t = sympy.Symbol("t")
dim = 4
q_funcs = [sympy.Function(f"q{i}") for i in range(1, 1 + dim)]
q_dot_funcs = [sympy.Function(f"qdot{i}") for i in range(1, 1 + dim)]
q_ddot_funcs = [sympy.Function(f"qddot{i}") for i in range(1, 1 + dim)]

# Putting this in a loop didn't work for god knows what reason
q_funcs[0].fdiff = lambda self, argindex=1: q_dot_funcs[0](self.args[argindex-1])
q_funcs[1].fdiff = lambda self, argindex=1: q_dot_funcs[1](self.args[argindex-1])
q_funcs[2].fdiff = lambda self, argindex=1: q_dot_funcs[2](self.args[argindex-1])
q_funcs[3].fdiff = lambda self, argindex=1: q_dot_funcs[3](self.args[argindex-1])

# Same with this
q_dot_funcs[0].fdiff = lambda self, argindex=1: q_ddot_funcs[0](self.args[argindex-1])
q_dot_funcs[1].fdiff = lambda self, argindex=1: q_ddot_funcs[1](self.args[argindex-1])
q_dot_funcs[2].fdiff = lambda self, argindex=1: q_ddot_funcs[2](self.args[argindex-1])
q_dot_funcs[3].fdiff = lambda self, argindex=1: q_ddot_funcs[3](self.args[argindex-1])

# Generalized coordinates
q = sympy.Matrix(
    [[q_funcs[i](t)] for i in range(dim)]
)
q1, q2, q3, q4 = q

# First derivatives
q_dot = sympy.diff(q, t)
q1_dot, q2_dot, q3_dot, q4_dot = q_dot

# Second derivatives
q_ddot = sympy.diff(q, t, t)
q1_ddot, q2_ddot, q3_ddot, q4_ddot = q_ddot

vars = [l1, l2, m1, m2, I1, I2, J1, J2, K1, K2, B1, B2, g, q, q_dot, q_ddot]

# Find the xy position of each center of mass
cm1_xy = sympy.Matrix([
        [-0.5 * l1 * sympy.sin(q2)],
        [0.5 * l1 * sympy.cos(q2)]
    ])
cm2_xy = sympy.Matrix([
    [-l1 * sympy.sin(q2) - 0.5 * l2 * sympy.sin(q2 + q4)],
    [l1 * sympy.cos(q2) + 0.5 * l2 * sympy.cos(q2 + q4)]
])

# Find the velocity vector for each center of mass
# sympy.diff(arg1, arg2) will be useful here (takes derivative of first argument with respect to the second)
cm1_xy_vel = sympy.diff(cm1_xy, t)
cm2_xy_vel = sympy.diff(cm2_xy, t)

# Now, we need to find the kinetic energy T
# This will be the sum of the translational kinetic energy and rotational kinetic energy

# Get the squared magnitude of the velocity for each center of mass for the translational kinetic energy
# Use the dot product to do this
cm1_vel_squared = cm1_xy_vel.dot(cm1_xy_vel)
cm2_vel_squared = cm2_xy_vel.dot(cm2_xy_vel)

# Get total kinetic energy
T_1 = 0.5 * I1 * q2_dot ** 2 + 0.5 * m1 * cm1_vel_squared
T_2 = 0.5 * I2 * (q2_dot + q4_dot) ** 2 + 0.5 * m2 * cm2_vel_squared
T_3 = 0.5 * J1 * q1_dot ** 2 # from motor 1
T_4 = 0.5 * J2 * q3_dot ** 2 # from motor 2
T = T_1 + T_2 + T_3 + T_4
T_func = sympy.lambdify(vars, T, "numpy")

# Next, we'll get the potential energy
V_1 = m1 * g * cm1_xy[1]
V_2 = m2 * g * cm2_xy[1]
V_3 = 0.5 * K1 * (q2 - q1) ** 2 # from motor 1
V_4 = 0.5 * K2 * (q4 - q3) ** 2 # from motor 2
V = V_1 + V_2 + V_3 + V_4
V_func = sympy.lambdify(vars, V, "numpy")

# Here's a helper function to do our Lagrangian Dynamics!
def LagrangianDynamics(T, V, q, dq):
    f_diff = sympy.diff(T, dq).T
    M = sympy.Matrix.hstack(*[sympy.diff(f_diff[i], dq) for i in range(dim)])
    C = sympy.zeros(dim, dim)
    # Don't ask me how this works
    for k in range(dim):
        for j in range(dim):
            for i in range(dim):
                C[k, j] = C[k, j] + (1/2)*(sympy.diff(M[k, j], q[i]) + sympy.diff(M[k, i], q[j]) - sympy.diff(M[i, j], q[k]))*dq[i]
    G = sympy.diff(V, q)
    return M, C, G

# We can now directly compute our dynamics
M, C, G = LagrangianDynamics(T, V, q, q_dot)

# Full equations of motion
LHS = M * q_ddot + C * q_dot + G
LHS_func = sympy.lambdify(vars, LHS, "numpy") # DO NOT MODIFY

# Now modify to add damping
Damping = sympy.Matrix([
    [B1, -B1, 0, 0],
    [-B1, B1, 0, 0],
    [0, 0, B2, -B2],
    [0, 0, -B2, B2]
    ])
LHS_Dampers = M * q_ddot + C * q_dot + G + Damping * q_dot
LHS_Dampers_func = sympy.lambdify(vars, LHS_Dampers, "sympy")

if __name__ == "__main__":
    print("\nXY Position of first center of mass:")
    sympy.pprint(cm1_xy)

    print("\nXY Position of second center of mass:")
    sympy.pprint(cm2_xy)

    print("\nTotal kinetic energy of system:")
    sympy.pprint(T)

    print("\nTotal potential energy of system:")
    sympy.pprint(V)