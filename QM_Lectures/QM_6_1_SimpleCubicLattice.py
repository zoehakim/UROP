import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Define lattice constants
a = 1.0  # lattice constant for the real-space lattice

# Real-space lattice points for a simple cubic lattice
lattice_points = np.array([[i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]) * a

# Define vertices for the real-space unit cell (a cube)
unit_cell_vertices = np.array([
    [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
    [0, 0, a], [a, 0, a], [a, a, a], [0, a, a]
])

# Define faces for the real-space unit cell (a cube)
unit_cell_faces = [
    [unit_cell_vertices[j] for j in [0, 1, 2, 3]],  # bottom face
    [unit_cell_vertices[j] for j in [4, 5, 6, 7]],  # top face
    [unit_cell_vertices[j] for j in [0, 1, 5, 4]],  # front face
    [unit_cell_vertices[j] for j in [2, 3, 7, 6]],  # back face
    [unit_cell_vertices[j] for j in [1, 2, 6, 5]],  # right face
    [unit_cell_vertices[j] for j in [0, 3, 7, 4]]   # left face
]




# Reciprocal lattice vectors for a simple cubic lattice
b1 = 2 * np.pi / a * np.array([1, 0, 0])
b2 = 2 * np.pi / a * np.array([0, 1, 0])
b3 = 2 * np.pi / a * np.array([0, 0, 1])

reciprocal_lattice_points = np.array([
    n * b1 + m * b2 + l * b3 for n in range(-1, 2) for m in range(-1, 2) for l in range(-1, 2)
])

# Define vertices for the Brillouin zone (a cube in reciprocal space)
bz_vertices = np.array([
    [0, 0, 0], [b1[0], 0, 0], [b1[0], b2[1], 0], [0, b2[1], 0],
    [0, 0, b3[2]], [b1[0], 0, b3[2]], [b1[0], b2[1], b3[2]], [0, b2[1], b3[2]]
])

# Define faces for the Brillouin zone (a cube)
bz_faces = [
    [bz_vertices[j] for j in [0, 1, 2, 3]],  # bottom face
    [bz_vertices[j] for j in [4, 5, 6, 7]],  # top face
    [bz_vertices[j] for j in [0, 1, 5, 4]],  # front face
    [bz_vertices[j] for j in [2, 3, 7, 6]],  # back face
    [bz_vertices[j] for j in [1, 2, 6, 5]],  # right face
    [bz_vertices[j] for j in [0, 3, 7, 4]]   # left face
]

# Define the high-symmetry points for the band structure
Γ = np.array([0, 0, 0])
X = np.array([np.pi/a, 0, 0])
M = np.array([np.pi/a, np.pi/a, 0])
R = np.array([np.pi/a, np.pi/a, np.pi/a])

k_path_points = [Γ, X, M, R]
k_path_lines = [[Γ, X], [X, M], [M, R]]

# Define a path in k-space for band structure: Γ -> X -> M -> R
k_path = np.concatenate([
    np.linspace(Γ, X, 100),
    np.linspace(X, M, 100),
    np.linspace(M, R, 100)
])





# Plot real-space lattice
fig = plt.figure()
ax1 = fig.add_subplot(projection='3d')
ax1.scatter(lattice_points[:, 0], lattice_points[:, 1], lattice_points[:, 2], color='b', s=50)
ax1.add_collection3d(Poly3DCollection(unit_cell_faces, color='cyan', alpha=0.3, edgecolor='k'))
ax1.set_title('Simple Cubic Lattice (Real Space)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_box_aspect([1,1,1])

# Plot reciprocal lattice
fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.scatter(reciprocal_lattice_points[:, 0], reciprocal_lattice_points[:, 1], reciprocal_lattice_points[:, 2], color='r', s=50)
ax2.add_collection3d(Poly3DCollection(bz_faces, color='orange', alpha=0.3, edgecolor='k'))

# Highlight the k-path in the Brillouin zone
for line in k_path_lines:
    ax2.plot(*zip(*line), color='purple', linewidth=2)

# Label high-symmetry points
for point, label in zip(k_path_points, ['Γ', 'X', 'M', 'R']):
    ax2.text(*point, label, color='black', fontsize=12, ha='center')

ax2.set_title('Reciprocal Lattice')
ax2.set_xlabel('k_x')
ax2.set_ylabel('k_y')
ax2.set_zlabel('k_z')
ax2.set_box_aspect([1,1,1])




# Compute non-interacting band structure
energies1 = k_path[:, 0] **2 + k_path[:, 1] **2 + k_path[:, 2] **2
energies21 = (b1[0] + k_path[:, 0]) **2 + (b1[1] + k_path[:, 1]) **2 + (b1[2] + k_path[:, 2]) **2
energies22 = (-b1[0] + k_path[:, 0]) **2 + (-b1[1] + k_path[:, 1]) **2 + (-b1[2] + k_path[:, 2]) **2
energies31 = (b2[0] + k_path[:, 0]) **2 + (b2[1] + k_path[:, 1]) **2 + (b2[2] + k_path[:, 2]) **2
energies32 = (-b2[0] + k_path[:, 0]) **2 + (-b2[1] + k_path[:, 1]) **2 + (-b2[2] + k_path[:, 2]) **2
energies41 = (b3[0] + k_path[:, 0]) **2 + (b3[1] + k_path[:, 1]) **2 + (b3[2] + k_path[:, 2]) **2
energies42 = (-b3[0] + k_path[:, 0]) **2 + (-b3[1] + k_path[:, 1]) **2 + (-b3[2] + k_path[:, 2]) **2
energies51 = (b1[0]-b2[0] + k_path[:, 0]) **2 + (b1[1]-b2[0] + k_path[:, 1]) **2 + (b1[2]-b2[0] + k_path[:, 2]) **2

# Plot band structure
plt.figure(figsize=(8, 6))
plt.plot(energies1, label="Non-Interacting Band Structure", color='b', lw=3)
plt.plot(energies21, label="Non-Interacting Band Structure", color='c')
plt.plot(energies22, label="Non-Interacting Band Structure", color='c')
plt.plot(energies31, label="Non-Interacting Band Structure", color='g')
plt.plot(energies32, label="Non-Interacting Band Structure", color='g')
plt.plot(energies41, label="Non-Interacting Band Structure", color='y')
plt.plot(energies42, label="Non-Interacting Band Structure", color='y')
plt.plot(energies51, label="Non-Interacting Band Structure", color='m')
plt.axvline(100, color='k', linestyle='--', label='X')
plt.axvline(200, color='k', linestyle='--', label='M')
plt.axvline(300, color='k', linestyle='--', label='R')
plt.xticks([0, 100, 200, 300], ['Γ', 'X', 'M', 'R'])
plt.ylabel("Energy (E)")
plt.xlabel("k-path")
plt.legend()
plt.title("Non-Interacting Band Structure for Simple Cubic Lattice")
plt.grid()
plt.show()
