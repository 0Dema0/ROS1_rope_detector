import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation as R

def draw_circle_3d(ax, center, normal, radius=0.05, color='black', resolution=20):
    theta = np.linspace(0, 2*np.pi, resolution)
    circle_pts = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])

    default_normal = np.array([0, 0, 1])
    if not np.allclose(normal, default_normal):
        rot, _ = R.align_vectors([normal], [default_normal])
        circle_pts = rot.apply(circle_pts.T).T

    circle_pts += np.reshape(center, (3,1))
    ax.plot(circle_pts[0], circle_pts[1], circle_pts[2], color=color)

def draw_simple_uav(ax, pos, orientation, scale=0.2, color='blue'):
    length = scale * 1.0  # body length (X)
    width = scale * 0.5   # body width (Y)
    height = scale * 0.25  # body height (Z)

    x = [-length/2, length/2]
    y = [-width/2, width/2]
    z = [-height/2, height/2]
    box = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z])

    faces = [
        [0,1,3,2], [4,5,7,6], [0,1,5,4],
        [2,3,7,6], [0,2,6,4], [1,3,7,5]
    ]

    rot = R.from_euler('xyz', orientation).as_matrix()
    box_world = (rot @ box.T).T + pos
    poly3d = [[box_world[i] for i in face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, edgecolors='k', linewidths=0.5, alpha=0.8))

    # Original arm start points (corners)
    arm_ends_local = np.array([
        [ length/2,  width/2, height/2],
        [-length/2,  width/2, height/2],
        [-length/2, -width/2, height/2],
        [ length/2, -width/2, height/2]
    ])

    # Move arm start points *inside* the body by a factor, e.g. 20% closer to center
    inset_factor = 0.2
    arm_starts_local = arm_ends_local * (1 - inset_factor)

    # Direction vectors for extending arms outward (XY plane)
    dir_vectors = arm_ends_local[:, :2]
    norms = np.linalg.norm(dir_vectors, axis=1, keepdims=True)
    dir_unit = dir_vectors / norms

    arm_length = scale * 0.5

    # Rotor centers are offset beyond arm ends
    rotor_centers_local = arm_ends_local.copy()
    for i in range(4):
        rotor_centers_local[i, 0] += dir_unit[i, 0] * arm_length
        rotor_centers_local[i, 1] += dir_unit[i, 1] * arm_length

    # Transform all to world coords
    arm_starts_world = (rot @ arm_starts_local.T).T + pos
    rotor_centers_world = (rot @ rotor_centers_local.T).T + pos

    # Draw arms from inset start points to rotor centers
    segments = [[start, end] for start, end in zip(arm_starts_world, rotor_centers_world)]
    arm_lines = Line3DCollection(segments, colors='black', linewidths=4, alpha=0.9)
    ax.add_collection3d(arm_lines)

    # Draw bigger rotors
    for r in rotor_centers_world:
        draw_circle_3d(ax, r, rot[:, 2], radius=scale * 0.35, color='black')

def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Test positions and orientations
    uav_pos = np.array([0, 0, 0.5])
    hook_pos = np.array([0.2, 0.1, 0.0])

    # Try different orientations here (in radians)
    orientation = [np.radians(10), np.radians(20), np.radians(45)]

    draw_simple_uav(ax, uav_pos, orientation)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.2, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Simple UAV and Hook Visualization")

    ax.view_init(elev=30, azim=45)  # Camera angle
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
