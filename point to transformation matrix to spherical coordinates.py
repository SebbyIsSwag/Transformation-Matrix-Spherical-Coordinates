!pip install numpy
!pip install scipy
import numpy as np
from scipy.spatial.transform import Rotation as R

# Generate a random 4D coordinate
x_cart = np.random.rand(4)

# Generate a random rotation quaternion
r = R.random(random_state=123).as_quat()
q = np.zeros((4, 4))
q[:3, :3] = R.from_quat(r).as_matrix()
q[3, 3] = 1

# Generate a random scaling factor
s = np.diag(np.random.rand(4))

# Generate a random translation vector
t = np.random.rand(3)

# Build the transformation matrix
T = np.zeros((4, 4))
T[:3, :3] = np.dot(q[:3, :3], s[:3, :3])
T[:3, 3] = t
T[3, 3] = 1

# Apply the transformation matrix to the point
x_prime_cart = np.dot(T, x_cart)

# Quaternion to spherical coordinates conversion function
def quat_to_sph(q):
    w, x, y, z = q
    r = np.sqrt(w**2 + x**2 + y**2 + z**2)
    theta = np.arccos(w / r)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    psi = np.arctan2(y, x)
    return np.array([r, theta, phi, psi])

# Convert the quaternion to spherical coordinates
q_sph = quat_to_sph(r)
