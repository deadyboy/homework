# Assignment 4 - Simplified 3D Gaussian Splatting

## 简介

本目录完成 DIP 课程 Homework4：使用 COLMAP 从多视角图像恢复相机和稀疏点云，并用纯 PyTorch 实现一个简化版 3D Gaussian Splatting renderer。主数据集使用课程提供的 `chair`。

参考作业目录：[YudongGuo/DIP-Teaching - Assignments/04_3DGS](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/04_3DGS)

## 文件说明

```text
homework4/
├── gaussian_model.py              # 3D Gaussian 参数、scale/rotation/covariance
├── gaussian_renderer.py           # 3D 到 2D 投影、Gaussian rasterization、alpha compositing
├── data_utils.py                  # COLMAP 文本模型与图像数据读取
├── mvs_with_colmap.py             # COLMAP SfM 流程
├── debug_mvs_by_projecting_pts.py # 点云重投影可视化
├── train.py                       # 简化 3DGS 训练脚本
├── render_3dgs_mv.py              # 训练后水平环绕视角渲染
├── smoke_test_3dgs.py             # 核心张量前向 smoke test
├── pics/                          # README 展示图
└── logs/run_summary.txt           # 运行摘要
```

## 环境与运行

本次实验使用独立 Python 环境运行。关键依赖包括 `COLMAP 3.13.0`、`PyTorch 2.11.0+cu128`、`OpenCV 4.13.0`、`natsort`、`tqdm`。注意 COLMAP 3.13 的 GPU 参数名是 `FeatureExtraction.use_gpu` 和 `FeatureMatching.use_gpu`，因此我更新了原脚本中的旧参数名。

```bash
conda create -n dip_hw4_3dgs python=3.10 colmap opencv natsort tqdm imageio imageio-ffmpeg ffmpeg numpy scipy -c conda-forge
conda activate dip_hw4_3dgs
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

运行顺序：

```bash
python mvs_with_colmap.py --data_dir data/chair
python debug_mvs_by_projecting_pts.py --data_dir data/chair
python smoke_test_3dgs.py
python train.py --colmap_dir data/chair \
  --checkpoint_dir outputs/chair_checkpoints \
  --num_epochs 200 \
  --debug_every 10 \
  --debug_samples 4 \
  --device cuda
```

## 实现说明

### 1. COLMAP SfM

`mvs_with_colmap.py` 依次运行 feature extraction、exhaustive matching、mapper 和 model converter，输出 `sparse/0_text` 下的 `cameras.txt`、`images.txt`、`points3D.txt`。本次 `chair` 数据集共注册 100 张图像，生成 13,458 个稀疏 3D 点。

重投影调试图如下，左侧是带重投影点的原图，右侧是稀疏点云在该视角下的投影效果：

<img src="pics/colmap_projection_r0.png" alt="COLMAP projection debug" width="700">

### 2. 3D Gaussian Model

`gaussian_model.py` 中实现了：

- 位置 `positions`：由 COLMAP 稀疏点初始化；
- 颜色 `colors`：由 COLMAP 点云 RGB 初始化，并用 logit 空间优化；
- 透明度 `opacities`：用 logit 空间优化；
- scale：用纯 PyTorch 的 `torch.cdist + topk` 估计近邻距离，不再依赖 PyTorch3D；
- covariance：由 quaternion 得到旋转矩阵 `R`，再计算 `R @ diag(scale^2) @ R.T`。

### 3. Gaussian Renderer

`gaussian_renderer.py` 中实现了：

- 世界坐标到相机坐标的变换；
- pinhole camera 投影；
- perspective projection Jacobian；
- 3D covariance 到 2D covariance 的投影；
- 2D Gaussian value；
- 按深度排序后的 front-to-back alpha compositing。

为了保持纯 PyTorch 版本可运行，我没有使用官方 3DGS 的 tile-based rasterizer、adaptive densification 或 CUDA fused kernel。因此本实现更适合作为教学版 baseline，渲染速度和质量都不能与官方实现直接等价。

## 实验结果

每张 debug 图上排是 GT 训练视角，下排是当前模型渲染结果。

**Epoch 0：**

<img src="pics/train_epoch_0000.png" alt="Training debug epoch 0" width="800">

**Epoch 100：**

<img src="pics/train_epoch_0100.png" alt="Training debug epoch 100" width="800">

**Epoch 190：**

<img src="pics/train_epoch_0190.png" alt="Training debug epoch 190" width="800">

训练 200 个 epoch 后，日志中的最后 loss 约为 `0.0408`。可以看到模型从初始模糊 Gaussian blob 逐渐学到了椅子的主体结构和绿色纹理，但由于没有 densification 和高效 rasterization，边缘仍有明显拖影，细节也弱于官方 3DGS。

## 验证

已完成的验证：

```text
COLMAP registered images: 100
COLMAP sparse points: 13458
Projection debug images: 100
Smoke test: SMOKE_OK cuda torch.Size([16, 16, 3])
Training epochs: 200
Final logged loss: 0.0408
Generated checkpoint: outputs/chair_checkpoints/checkpoint_000180.pt
```

官方 3DGS 对比没有在本次提交中完整复现，原因是官方实现需要额外 CUDA extension / rasterizer 环境；本 README 在实现说明和结果分析中已明确说明简化版与官方版的主要差异。
