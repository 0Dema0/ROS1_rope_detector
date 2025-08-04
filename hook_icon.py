import numpy as np
import matplotlib.pyplot as plt

def create_single_hook(length_straight=15, bend_radius=7.5, n_straight=50, n_bend=100, n_tip=30, barb_length=1):
    # 1. Straight segment along Z
    z_straight = np.linspace(0, length_straight, n_straight)
    x_straight = np.zeros_like(z_straight)
    y_straight = np.zeros_like(z_straight)

    # 2. U-bend in YZ plane
    theta_bend = np.linspace(0, np.pi, n_bend)
    z_bend = length_straight + bend_radius * np.sin(theta_bend)
    x_bend = np.zeros_like(z_bend)
    y_bend = bend_radius * (1 - np.cos(theta_bend))

    # 3. Tip extension (slight continuation)
    theta_tip = np.linspace(np.pi, np.pi + 0.4, n_tip)
    z_tip = length_straight + bend_radius * np.sin(theta_tip)
    x_tip = np.zeros_like(z_tip)
    y_tip = bend_radius * (1 - np.cos(theta_tip))

    # 4. Barb segment (sharp tip)
    dx = x_tip[-1] - x_tip[-2]
    dy = y_tip[-1] - y_tip[-2]
    dz = z_tip[-1] - z_tip[-2]
    direction = np.array([dx, dy, dz])
    direction /= np.linalg.norm(direction)

    barb_direction = -direction + np.array([0, -1, 0])  # backward and downward
    barb_direction /= np.linalg.norm(barb_direction)
    barb_vector = barb_length * barb_direction

    x_barb = np.array([x_tip[-1], x_tip[-1] + barb_vector[0]])
    y_barb = np.array([y_tip[-1], y_tip[-1] + barb_vector[1]])
    z_barb = np.array([z_tip[-1], z_tip[-1] + barb_vector[2]])

    # Combine all parts
    x = np.concatenate([x_straight, x_bend, x_tip, x_barb])
    y = np.concatenate([y_straight, y_bend, y_tip, y_barb])
    z = np.concatenate([z_straight, z_bend, z_tip, z_barb])

    z = -z  # flip hook upside down along Z axis

    return np.vstack([x, y, z]).T  # Nx3 array

def rotate_z(points, angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    return points @ R.T

def create_four_hooks():
    base_hook = create_single_hook()
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    all_hooks = []
    for angle in angles:
        all_hooks.append(rotate_z(base_hook, angle))
    return all_hooks  # List of 4 Nx3 arrays

def rotation_matrix_from_vectors(vec1, vec2):
    """ Returns rotation matrix to rotate vec1 to vec2 """
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c == -1:
        # 180 degree rotation: find orthogonal vector for rotation axis
        orth = np.array([1,0,0]) if (abs(a[0]) < 0.9) else np.array([0,1,0])
        v = np.cross(a, orth)
        v /= np.linalg.norm(v)
        H = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = -np.eye(3) + 2 * np.outer(v, v)
        return R
    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R

def transform_hook(hook_points, rotation_matrix, translation_vector):
    """Rotate and translate hook points"""
    rotated = hook_points @ rotation_matrix.T
    translated = rotated + translation_vector
    return translated

def plot_four_hooks(hooks, color='red'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for hook in hooks:
        ax.plot(hook[:, 0], hook[:, 1], hook[:, 2], color=color, linewidth=6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    plt.show()

# Example usage:
if __name__ == '__main__':
    hooks = create_four_hooks()
    plot_four_hooks(hooks, color='red')