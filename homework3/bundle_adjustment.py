"""
Task 1: Bundle Adjustment with PyTorch
从 2D 观测恢复 3D 点云、相机参数和焦距
"""

import numpy as np
import torch
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import os
import math

np.random.seed(0)
torch.manual_seed(0)

# ============== Euler 角转旋转矩阵 ==============
def euler_angles_to_matrix_xyz(euler_angles):
    """
    将 Euler 角 (XYZ 顺序) 转换为旋转矩阵

    Args:
        euler_angles: (*, 3) Euler 角 (rx, ry, rz) 弧度制

    Returns:
        rotation matrices: (*, 3, 3)
    """
    shape = euler_angles.shape[:-1]
    ea = euler_angles.reshape(-1, 3)

    rx, ry, rz = ea[:, 0], ea[:, 1], ea[:, 2]

    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    R_x = torch.stack([
        torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx),
        torch.zeros_like(rx), cos_x, -sin_x,
        torch.zeros_like(rx), sin_x, cos_x
    ], dim=-1).reshape(-1, 3, 3)

    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    R_y = torch.stack([
        cos_y, torch.zeros_like(ry), sin_y,
        torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry),
        -sin_y, torch.zeros_like(ry), cos_y
    ], dim=-1).reshape(-1, 3, 3)

    cos_z, sin_z = torch.cos(rz), torch.sin(rz)
    R_z = torch.stack([
        cos_z, -sin_z, torch.zeros_like(rz),
        sin_z, cos_z, torch.zeros_like(rz),
        torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)
    ], dim=-1).reshape(-1, 3, 3)

    R = R_z @ R_y @ R_x
    return R.reshape(*shape, 3, 3)


# ============== 配置 ==============
IMAGE_SIZE = 1024
NUM_VIEWS = 50
NUM_POINTS = 20000
DATA_DIR = "data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============== 数据加载 ==============
print("Loading data...")
points2d_data = np.load(f"{DATA_DIR}/points2d.npz")
points3d_colors = np.load(f"{DATA_DIR}/points3d_colors.npy")

observations = np.zeros((NUM_VIEWS, NUM_POINTS, 2), dtype=np.float32)
visibilities = np.zeros((NUM_VIEWS, NUM_POINTS), dtype=np.float32)

for i in range(NUM_VIEWS):
    key = f"view_{i:03d}"
    obs = points2d_data[key]
    observations[i] = obs[:, :2]
    visibilities[i] = obs[:, 2]

observations = torch.tensor(observations, dtype=torch.float32, device=device)
visibilities = torch.tensor(visibilities, dtype=torch.float32, device=device)

total_visible = int(visibilities.sum().item())
print(f"Observations shape: {observations.shape}")
print(f"Total visible observations: {total_visible}")

# ============== 参数初始化 ==============
print("\nInitializing parameters...")

# 焦距初始化：f = H / (2 * tan(fov/2))
fov_deg = 60  # 视场角 60°
focal_init = IMAGE_SIZE / (2 * np.tan(np.deg2rad(fov_deg / 2)))
# 用 log 参数化保证焦距恒为正
focal_log = nn.Parameter(torch.tensor([math.log(focal_init)], dtype=torch.float32, device=device))

# 相机外参
euler_angles = nn.Parameter(torch.zeros(NUM_VIEWS, 3, dtype=torch.float32, device=device))

# 平移：[0, 0, -d]
d_init = 2.5
translations = nn.Parameter(torch.zeros(NUM_VIEWS, 3, dtype=torch.float32, device=device))
translations.data[:, 2] = -d_init

# 3D 点：原点附近随机
points3d = nn.Parameter(torch.randn(NUM_POINTS, 3, dtype=torch.float32, device=device) * 0.1)

print(f"Focal length init: {focal_init:.2f} (FoV={fov_deg}°, f = exp(log_f))")
print(f"Translation init: [0, 0, {-d_init}]")
print(f"3D points init: random in [-0.1, 0.1]^3")

# ============== 投影函数 ==============
def project(points3d, euler_angles, translations, focal):
    """
    将 3D 点投影到 2D 图像平面

    相机变换: [Xc, Yc, Zc] = R @ P + T
    投影: u = -f * Xc/Zc + cx, v = f * Yc/Zc + cy
    """
    V = euler_angles.shape[0]
    N = points3d.shape[0]

    R = euler_angles_to_matrix_xyz(euler_angles)  # (V, 3, 3)

    points3d_exp = points3d.unsqueeze(0).expand(V, -1, -1)  # (V, N, 3)
    rotated = torch.bmm(R, points3d_exp.transpose(1, 2)).transpose(1, 2)  # (V, N, 3)
    camera_points = rotated + translations.unsqueeze(1)  # (V, N, 3)

    Xc = camera_points[:, :, 0]
    Yc = camera_points[:, :, 1]
    Zc = camera_points[:, :, 2]

    cx, cy = IMAGE_SIZE / 2, IMAGE_SIZE / 2
    u = -focal * Xc / Zc + cx
    v = focal * Yc / Zc + cy

    projected = torch.stack([u, v], dim=-1)
    return projected, Zc


# ============== 损失函数 ==============
def compute_loss(projected, observations, visibilities,
                 points3d, euler_angles, translations, cam_z):
    """L2 重投影误差（MSE，只计算可见点）+ 正则化"""
    # 数据损失：重投影误差
    diff = projected - observations  # (V, N, 2)
    sq_dist = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2  # (V, N)
    data_loss = (sq_dist * visibilities).sum() / (visibilities.sum() + 1e-8)

    # 正则化：保持 3D 点中心在原点附近
    center_reg = points3d.mean(dim=0).pow(2).sum()
    # 正则化：防止相机跑到物体后面 (Zc 应为负)
    z_front_penalty = torch.relu(cam_z + 1e-2).mean()
    # 正则化：防止位姿参数爆炸
    pose_reg = 1e-4 * (euler_angles.pow(2).mean() + translations.pow(2).mean())

    loss = data_loss + 1e-3 * center_reg + 1e-2 * z_front_penalty + pose_reg
    return loss, data_loss


# ============== 优化 ==============
params = [
    {'params': [focal_log], 'lr': 0.01},
    {'params': [euler_angles], 'lr': 0.01},
    {'params': [translations], 'lr': 0.01},
    {'params': [points3d], 'lr': 0.1},
]

optimizer = optim.Adam(params)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)

print("\nStarting optimization...")
num_iterations = 3000
log_interval = 200

loss_history = []
best_loss = float('inf')
best_params = None

for iteration in range(num_iterations):
    optimizer.zero_grad()
    focal = torch.exp(focal_log)
    projected, cam_z = project(points3d, euler_angles, translations, focal)
    loss, data_loss = compute_loss(projected, observations, visibilities,
                                   points3d, euler_angles, translations, cam_z)
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = {
            'focal': torch.exp(focal_log).detach().cpu().numpy().copy(),
            'euler_angles': euler_angles.detach().cpu().numpy().copy(),
            'translations': translations.detach().cpu().numpy().copy(),
            'points3d': points3d.detach().cpu().numpy().copy(),
        }

    if iteration % log_interval == 0 or iteration == num_iterations - 1:
        lr = optimizer.param_groups[0]['lr']
        data_loss_val = data_loss.item()
        print(f"Iter {iteration:4d}: Loss = {loss.item():.6f} (data={data_loss_val:.4f}), LR = {lr:.6f}")

print(f"\nOptimization finished! Best loss: {best_loss:.6f}")

# ============== Loss 曲线 ==============
plt.figure(figsize=(10, 6))
plt.plot(loss_history, linewidth=1.5)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Mean Squared Reprojection Error (px^2)', fontsize=14)
plt.title('Bundle Adjustment - Loss Curve', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
print("Saved loss_curve.png")

# ============== 3D 点云 (OBJ 格式) ==============
points3d_final = best_params['points3d']
colors = points3d_colors  # 数据中的颜色值

with open('reconstruction.obj', 'w') as f:
    for i in range(NUM_POINTS):
        x, y, z = points3d_final[i]
        r, g, b = colors[i]
        f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.3f} {g:.3f} {b:.3f}\n")

print(f"Saved reconstruction.obj ({NUM_POINTS} points)")

# ============== 相机参数 ==============
print("\n=== Camera Parameters ===")
print(f"Focal length: {best_params['focal'][0]:.4f}")
print("\nCamera rotations (Euler angles in degrees):")
euler_deg = np.degrees(best_params['euler_angles'])
for i in [0, 12, 25, 37, 49]:
    print(f"  Camera {i:2d}: rx={euler_deg[i,0]:7.2f}°, ry={euler_deg[i,1]:7.2f}°, rz={euler_deg[i,2]:7.2f}°")
print("\nCamera translations:")
for i in [0, 12, 25, 37, 49]:
    print(f"  Camera {i:2d}: [{best_params['translations'][i,0]:6.3f}, {best_params['translations'][i,1]:6.3f}, {best_params['translations'][i,2]:6.3f}]")

# ============== 验证 ==============
print("\n=== Verification ===")
best_focal = torch.tensor(best_params['focal'], device=device)
euler_t = torch.tensor(best_params['euler_angles'], device=device)
trans_t = torch.tensor(best_params['translations'], device=device)
pts_t = torch.tensor(best_params['points3d'], device=device)

with torch.no_grad():
    proj_final, _ = project(pts_t, euler_t, trans_t, best_focal)
    diff = proj_final - observations
    sq_dist = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2
    mse = (sq_dist * visibilities).sum() / visibilities.sum()
    rmse = torch.sqrt(mse)
    err = torch.sqrt(sq_dist)
    visible_err = err[visibilities > 0]
    print(f"Mean reprojection error (visible): RMSE={rmse.item():.4f} px, MSE={mse.item():.4f}")
    print(f"Max reprojection error (visible):  {visible_err.max().item():.4f} pixels")

print("\nDone!")
