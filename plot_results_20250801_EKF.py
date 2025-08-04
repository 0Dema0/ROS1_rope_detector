import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d
from datetime import datetime

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
    hook = np.vstack((-gt_hook_x, -gt_hook_y, gt_hook_z)).T
    uav = np.vstack((-gt_uav_x, -gt_uav_y, gt_uav_z-150)).T
    gt = (hook - uav) / 1000.0  # Convert from mm to meters

    return t, gt

def plot_est_gt(t_est, est, ekf, seen_front, seen_rear, t_gt, gt):
    # Find the global earliest timestamp across both datasets
    global_start = t_gt.min() #min(t_est.min(), t_gt.min())

    # Shift timestamps to start at zero relative to global_start
    t_est_shifted = t_est - global_start
    t_gt_shifted = t_gt - global_start

    plt.figure(figsize=(12, 10))

    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = plt.subplot(4, 1, i + 1)
        ax.plot(t_est_shifted, est[:, i], 'r--', label='Estimated' if i == 0 else "")
        ax.plot(t_gt_shifted, gt[:, i], 'b-', label='Ground Truth' if i == 0 else "")
        ax.plot(t_est_shifted, ekf[:, i], 'g:', label='EKF' if i == 0 else "")
        ax.set_xlim(0, 30)
        ax.set_ylabel(f'{label} [m]')
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right')

    ax4 = plt.subplot(4, 1, 4)
    ax4.step(t_est_shifted, seen_front, where='post', label='Front Camera', color='orange', linestyle='--')
    ax4.step(t_est_shifted, seen_rear, where='post', label='Rear Camera', color='purple', linestyle=':')
    ax4.set_xlim(0, 30)
    ax4.set_ylabel('Seen')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.set_xlabel('Time [s]')
    ax4.grid(True)
    ax4.legend(loc='upper right')

    plt.suptitle('Hook Position in Base Link Frame')
    plt.tight_layout()
    plt.show()

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
    t_gt, gt = read_qualisys_tsv(tsv_file)

    print("Estimator start time:", t_est[0])
    print("Qualisys start time:", t_gt[0])

    rmse_est, rmse_ekf = compute_rmse_from_aligned(t_gt, gt, t_est, est, ekf)

    print("RMSE (GT vs EST):", rmse_est)
    print("RMSE (GT vs EKF):", rmse_ekf)

    plot_est_gt(t_est, est, ekf, seen_front, seen_rear, t_gt, gt)
