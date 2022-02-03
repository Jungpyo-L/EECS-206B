import sympy

# Use this file to implement the forward kinematics and jacobian map for the SCAR manipulator
# Reccomendation: Use sympy.pprint(...) for displaying any symbolic expressions

# Setup Symbolic Variables
theta = sympy.symbols("theta1:5")
L = sympy.symbols("l0:3")
vars = theta + L

# Other constants
q1 = sympy.zeros(3, 1)
q2 = sympy.Matrix([
    [0],
    [L[1]],
    [0]
])
q3 = sympy.Matrix([
    [0],
    [L[1] + L[2]],
    [L[0]]
])

N_dof = 4

# Define twists
# sympy.Matrix, sympy.zeros may be useful. SymPy also implemented a cross product for you (vec1.cross(vec2))
omega = [... for _ in range(N_dof)]
v = [... for _ in range(N_dof)]
omega_prime = [... for _ in range(N_dof)]
v_prime = [... for _ in range(N_dof)]

omega[0] = sympy.Matrix([
    [0],
    [0],
    [1]
])
v[0] = -omega[0].cross(q1)

omega[1] = sympy.Matrix([
    [0],
    [0],
    [1]
])
v[1] = -omega[1].cross(q2)

omega[2] = sympy.Matrix([
    [0],
    [0],
    [1]
])
v[2] = -omega[2].cross(q3)

omega[3] = sympy.Matrix([
    [0],
    [0],
    [0]
])
v[3] = sympy.Matrix([
    [0],
    [0],
    [1]
])

# Stack v and omega vectors into xi twist vectors
xi = [sympy.Matrix.vstack(v[i], omega[i]) for i in range(N_dof)]
print("Xi Vectors:")
sympy.pprint(xi)

# Feel free to define any helper functions!
# JP: define
def hat(p):
    hat = sympy.Matrix([
    [0, -p[2], p[1]],
    [p[2], 0, -p[0]],
    [-p[1], p[0], 0],
    ])
    return hat

def xi_hat(xi):
    xi_hat = sympy.Matrix([
    [0, -xi[5], xi[4], xi[0]],
    [xi[5], 0, -xi[3], xi[1]],
    [-xi[4], xi[3], 0, xi[2]],
    [0, 0, 0, 0]
    ])
    return xi_hat

def Ad(g):
    R = g[0:3, 0:3]
    p = g[0:3, 3]
    Ad = R
    Ad = Ad.col_insert(3, hat(p)*R)
    Ad = Ad.row_insert(3, R.col_insert(0, sympy.zeros(3, 3)))
    return Ad

def Ad_inv(g):
    R = g[0:3, 0:3]
    p = g[0:3, 3]
    R_t = R.T
    Ad = R_t
    Ad = Ad.col_insert(3, -R_t * hat(p))
    Ad = Ad.row_insert(3, R_t.col_insert(0, sympy.zeros(3, 3)))
    return Ad

def Rx(theta):
    Rx =  sympy.Matrix([
    [1, 0, 0],
    [0, sympy.cos(theta), -sympy.sin(theta)],
    [0, sympy.sin(theta), sympy.cos(theta)]
    ])
    return Rx

def Ry(theta):
    Rx =  sympy.Matrix([
    [sympy.cos(theta), 0, sympy.sin(theta)],
    [0, 1, 0],
    [-sympy.sin(theta), 0, sympy.cos(theta)]
    ])
    return Rx

def Rz(theta):
    Rx =  sympy.Matrix([
    [sympy.cos(theta), -sympy.sin(theta), 0],
    [sympy.sin(theta), sympy.cos(theta), 0],
    [0, 0, 1]
    ])
    return Rx

# Define gst(0)
gst_0 = sympy.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, L[1]+L[2]],
    [0, 0, 1, L[0]],
    [0, 0, 0, 1]
])

# (a) Calculate the forward kinematics map as a function of theta
e1 = xi_hat(xi[0])*theta[0]
e2 = xi_hat(xi[1])*theta[1]
e3 = xi_hat(xi[2])*theta[2]
e4 = xi_hat(xi[3])*theta[3]

g1 = sympy.exp(e1)
g1 = sympy.simplify(g1)
sympy.pprint(g1)
sympy.pprint(Ad_inv(g1))

g2 = sympy.exp(e2)
g2 = sympy.simplify(g2)
sympy.pprint(g2)
sympy.pprint(Ad_inv(g2))

g3 = sympy.exp(e3)
g3 = sympy.simplify(g3)

g4 = sympy.exp(e4)
g4 = sympy.simplify(g4)

gst = g1 * g2 * g3 * g4 * gst_0
#gst = sympy.simplify(gst)

print("Computed Forward Kinematics!")
gst_func = sympy.lambdify(vars, gst, "numpy")

# (b) Calculate the the spatial and body jacobians as a function of theta
"""
omega_prime[0] = omega[0]
omega_prime[1] = omega[1]
omega_prime[2] = omega[2]
omega_prime[3] = omega[3]
q1_prime = q1
q2_prime = Rz(theta[0]) * q2
q3_prime = q2_prime + Rz(theta[0]) * Rz(theta[1]) * (q3 - q2)
v_prime[0] = -omega_prime[0].cross(q1_prime)
v_prime[0] = sympy.simplify(v_prime[0])
v_prime[1] = -omega_prime[1].cross(q2_prime)
v_prime[1] = sympy.simplify(v_prime[1])
v_prime[2] = -omega_prime[2].cross(q3_prime)
v_prime[2] = sympy.simplify(v_prime[2])
v_prime[3] = v[3]
xi_prime = [sympy.Matrix.vstack(v_prime[i], omega_prime[i]) for i in range(N_dof)]
J_spatial = sympy.Matrix([xi_prime])

"""
# calculate with Ad function
xi_prime = xi
xi_prime[1] = Ad(g1) * xi[1]
xi_prime[1] = sympy.simplify(xi_prime[1])
xi_prime[2] = Ad(g1 * g2) * xi[2]
xi_prime[2] = sympy.simplify(xi_prime[2])
xi_prime[3] = Ad(g1 * g2 * g3) * xi[3]
xi_prime[3] = sympy.simplify(xi_prime[3])
J_spatial = sympy.Matrix([xi_prime])

#sympy.pprint(J_spatial)
print("Computed Spatial Jacobian!")
J_spatial_func = sympy.lambdify(vars, J_spatial, "numpy")

J_body = Ad_inv(gst) * J_spatial

"""
# body jacobian with analytical method
xi_dagger = xi
xi_dagger[0] = Ad_inv(g1 * gst_0) * xi[0]
xi_dagger[1] = Ad_inv(g1 * g2 * gst_0) * xi[1]
xi_dagger[2] = Ad_inv(g1 * g2 * g3 * gst_0) * xi[2]
xi_dagger[3] = Ad_inv(g1 * g2 * g3 * g4 * gst_0) * xi[3]

J_body = sympy.Matrix([xi_dagger])
"""
#sympy.pprint(J_body)

print("Computed Body Jacobian!")
J_body_func = sympy.lambdify(vars, J_body, "numpy")