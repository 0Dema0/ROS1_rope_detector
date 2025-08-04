import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation as R
from uav_icon import draw_simple_uav
from hook_icon import create_single_hook, rotation_matrix_from_vectors, transform_hook


start_time = 10
end_time = 11

def read_estimation_npz(npz_path):
    """
    Reads a .npz file from the estimator and returns:
      - timestamps (Unix time)
      - estimated positions in base_link frame (Nx3)
    """
    data = np.load(npz_path)
    t = data['timestamps']
    est = data['estimated']
    ekf = data['ekf']
    seen_front = data['seen_front_f']
    seen_rear = data['seen_rear_f']

    print('Data shape:', t.shape)

    return t, est, ekf, seen_front, seen_rear

def read_qualisys_tsv(tsv_path):
    """
    Reads a .tsv file from Qualisys and returns:
      - timestamps (Unix time, one per frame)
      - relative positions (hook - UAV), shape: (N, 3), in meters
    """
    # Step 1: Extract TIME_STAMP line
    with open(tsv_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("TIME_STAMP"):
            parts = line.strip().split('\t')
            time_str = parts[1].replace(',', '')  # e.g., "2025-07-29 17:28:46.346"
            base_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
            base_offset = float(parts[2])  # seconds
            break
    else:
        raise ValueError("TIME_STAMP not found in TSV file")

    # Step 2: Load data skipping headers
    data = np.genfromtxt(tsv_path, delimiter='\t', skip_header=14)
    num_samples = data.shape[0]

    # Step 3: Compute timestamps
    t = base_time + np.arange(num_samples) / 150.0  # 150 Hz

    # Step 4: Extract positions and convert to meters
    gt_hook_x = data[:, 2]
    gt_hook_y = data[:, 3]
    gt_hook_z = data[:, 4]
    gt_uav_x = data[:, 19]
    gt_uav_y = data[:, 20]
    gt_uav_z = data[:, 21]
    hook_gt = np.vstack((-gt_hook_x, -gt_hook_y, gt_hook_z)).T
    uav_gt = np.vstack((-gt_uav_x, -gt_uav_y, gt_uav_z-150)).T
    gt = (hook_gt - uav_gt) / 1000.0  # Convert from mm to meters

    return t, hook_gt / 1000.0, uav_gt / 1000.0, gt

def plot_est_gt(t_est, est, ekf, seen_front, seen_rear, t_gt, hook_gt, uav_gt, gt):
    # Find the global earliest timestamp across both datasets
    global_start = t_gt.min() #min(t_est.min(), t_gt.min())

    # Shift timestamps to start at zero relative to global_start
    # t_est_shifted = t_est - global_start
    t_gt_shifted = t_gt - global_start

    gt_mask = (t_gt_shifted >= start_time) & (t_gt_shifted <= end_time)

    hook_gt_filtered = hook_gt[gt_mask]
    uav_gt_filtered = uav_gt[gt_mask]
    time_filtered = t_gt_shifted[gt_mask]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(hook_gt_filtered[:, 0], hook_gt_filtered[:, 1], hook_gt_filtered[:, 2], 'r--', label='Hook Ground Truth')
    ax.plot(uav_gt_filtered[:, 0], uav_gt_filtered[:, 1], uav_gt_filtered[:, 2], 'b-', label='UAV Ground Truth')

    norm = mcolors.Normalize(vmin=time_filtered.min(), vmax=time_filtered.max())
    cmap = cm.get_cmap('viridis')

    # Build line segments for each timestamp pair
    segments = []
    colors = []

    for hook, uav, t in zip(hook_gt_filtered, uav_gt_filtered, time_filtered):
        segment = [hook, uav]
        segments.append(segment)
        colors.append(cmap(norm(t)))

    # Create the collection of 3D lines with gradient color
    lc = Line3DCollection(segments, colors=colors, linewidths=0.5, alpha=0.2)
    ax.add_collection3d(lc)

    scale = 0.001
    base_hook = create_single_hook() * scale

    # Pick 5 evenly spaced indices from hook_gt_filtered
    num_hooks = 4
    indices = np.linspace(0, len(hook_gt_filtered)-1, num_hooks, dtype=int)

    for idx in indices:
        p_hook = hook_gt_filtered[idx]
        p_uav = uav_gt_filtered[idx]

        # Z axis (pointing down toward hook)
        z_axis = p_hook - p_uav
        z_axis /= np.linalg.norm(z_axis)

        # X axis (UAV forward direction)
        if idx < len(uav_gt_filtered) - 1:
            x_axis = uav_gt_filtered[idx + 1] - p_uav
        else:
            x_axis = p_uav - uav_gt_filtered[idx - 1]
        x_axis /= np.linalg.norm(x_axis)

        # Make X orthogonal to Z (in case of drift)
        x_axis -= np.dot(x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)

        # Y axis = Z × X
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # Final rotation matrix (columns are body axes)
        R_uav = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3

        # Convert to Euler angles (XYZ order)
        euler_angles = R.from_matrix(R_uav).as_euler('xyz')

        # Draw the UAV
        draw_simple_uav(ax, p_uav, euler_angles, scale=0.01, color='blue')

        ax.plot([p_hook[0], p_uav[0]],
            [p_hook[1], p_uav[1]],
            [p_hook[2], p_uav[2]],
            color='red', linewidth=1, alpha=1.0)

        direction = p_uav - p_hook
        if np.linalg.norm(direction) < 1e-6:
            continue
        direction /= np.linalg.norm(direction)

        R_align = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)

        # Add extra twist around the direction vector
        twist_angle = np.radians(30)  # Change this to any angle you want to rotate the hooks around their own axis
        R_twist = rotate_around_vector(direction, twist_angle)

        # Generate 4 hook faces by rotating base around Z *before* aligning
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            R_local = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0,              0,             1]
            ])
            hook_rotated = base_hook @ R_local.T
            hook_twisted = hook_rotated @ R_twist.T
            transformed_hook = transform_hook(hook_twisted, R_align, p_hook)
            ax.plot(transformed_hook[:, 0], transformed_hook[:, 1], transformed_hook[:, 2], color='red', linewidth=1)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Hook Position in Map Frame')
    ax.legend()
    ax.grid(True)

    set_axes_equal(ax)

    plt.suptitle('Start: {} s   End: {} s'.format(start_time, end_time))
    plt.tight_layout()
    plt.show()

def rotate_around_vector(vec, angle):
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C    ]
    ])
    return R

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so objects are not distorted."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def interpolate_gt_to_est(t_gt, gt, t_est):
    """
    Interpolates ground truth positions to match estimator timestamps.

    Parameters:
        t_gt (np.ndarray): Ground truth timestamps (shape: N,)
        gt (np.ndarray): Ground truth positions (shape: N, 3)
        t_est (np.ndarray): Estimator timestamps to interpolate to (shape: M,)

    Returns:
        gt_interp (np.ndarray): Interpolated GT positions at t_est times (shape: M, 3)
    """
    interp_func = interp1d(t_gt, gt, axis=0, bounds_error=False, fill_value="extrapolate")
    gt_interp = interp_func(t_est)
    return gt_interp

def compute_rmse_from_aligned(t_gt, gt, t_est, est, ekf):
    """
    Computes RMSE for est and ekf against interpolated GT.

    Parameters:
        t_gt (np.ndarray): Ground truth timestamps (N,)
        gt (np.ndarray): Ground truth positions (N, 3)
        t_est (np.ndarray): Estimation timestamps (M,)
        est (np.ndarray): Estimated positions (M, 3)
        ekf (np.ndarray): EKF positions (M, 3)

    Returns:
        rmse_est (np.ndarray): RMSE between interpolated GT and EST (3,)
        rmse_ekf (np.ndarray): RMSE between interpolated GT and EKF (3,)
    """
    # Find overlapping time range
    t_start = max(t_gt[0], t_est[0])
    t_end   = min(t_gt[-1], t_est[-1])

    # Mask both datasets to common time window
    gt_mask = (t_gt >= t_start) & (t_gt <= t_end)
    est_mask = (t_est >= t_start) & (t_est <= t_end)

    t_gt_sync = t_gt[gt_mask]
    gt_sync   = gt[gt_mask]

    t_est_sync = t_est[est_mask]
    est_sync   = est[est_mask]
    ekf_sync   = ekf[est_mask]

    # Interpolate GT to EST timestamps
    gt_interp = interpolate_gt_to_est(t_gt_sync, gt_sync, t_est_sync)

    # Remove rows with NaNs
    valid_mask = ~np.isnan(gt_interp).any(axis=1) & \
                 ~np.isnan(est_sync).any(axis=1) & \
                 ~np.isnan(ekf_sync).any(axis=1)

    gt_valid  = gt_interp[valid_mask]
    est_valid = est_sync[valid_mask]
    ekf_valid = ekf_sync[valid_mask]

    # Compute RMSE
    rmse_est = np.sqrt(np.mean((gt_valid - est_valid) ** 2, axis=0))
    rmse_ekf = np.sqrt(np.mean((gt_valid - ekf_valid) ** 2, axis=0))

    return rmse_est, rmse_ekf

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python your_script.py <estimation_npz_file> <qualisys_tsv_file>")
        sys.exit(1)

    npz_file = sys.argv[1]
    tsv_file = sys.argv[2]

    t_est, est, ekf, seen_front, seen_rear = read_estimation_npz(npz_file)
    t_gt, hook_gt, uav_gt, gt = read_qualisys_tsv(tsv_file)

    print("Estimator start time:", t_est[0])
    print("Qualisys start time:", t_gt[0])

    rmse_est, rmse_ekf = compute_rmse_from_aligned(t_gt, gt, t_est, est, ekf)

    print("RMSE (GT vs EST):", rmse_est)
    print("RMSE (GT vs EKF):", rmse_ekf)

    plot_est_gt(t_est, est, ekf, seen_front, seen_rear, t_gt, hook_gt, uav_gt, gt)
