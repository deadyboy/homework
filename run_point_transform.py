import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation using MLS Similarity Transform
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    基于 MLS (Moving Least Squares) 相似变换的图像变形。

    采用逆向映射：对输出图像中的每个像素 v，求其在原图中的对应坐标。
    正向变换定义为 p_i (source) -> q_i (target)。

    Return
    ------
        A deformed image.
    """
    if len(source_pts) < 2 or len(source_pts) != len(target_pts):
        return np.array(image)

    img_array = np.array(image)
    h, w = img_array.shape[:2]
    n = len(source_pts)

    source_pts = source_pts.astype(np.float64)
    target_pts = target_pts.astype(np.float64)

    # Step 1: 生成输出图像全部像素坐标 v, shape (H*W, 2)
    yi, xi = np.indices((h, w))
    v = np.stack((xi, yi), axis=-1).reshape(-1, 2).astype(np.float64)  # (HW, 2)
    num_pixels = v.shape[0]

    # Step 2: 计算权重 w_i(v) = 1 / |q_i - v|^{2*alpha}
    # q_i 是目标控制点 (输出图像中的已知位置)
    # diff: (N, HW, 2)
    diff = target_pts[:, np.newaxis, :] - v[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)  # (N, HW)
    weights = 1.0 / (dist_sq ** alpha + eps)  # (N, HW)

    # Step 3: 加权质心
    # q* = sum(w_i * q_i) / sum(w_i)
    # p* = sum(w_i * p_i) / sum(w_i)
    sum_w = np.sum(weights, axis=0)  # (HW,)
    q_star = np.einsum('ij,ik->jk', weights, target_pts) / sum_w[:, np.newaxis]  # (HW, 2)
    p_star = np.einsum('ij,ik->jk', weights, source_pts) / sum_w[:, np.newaxis]  # (HW, 2)

    # 这里 np.einsum('ij,ik->jk', weights, pts) 等价于:
    #   对每个像素 j: sum_i weights[i,j] * pts[i,:],  即按列(像素)做加权求和

    # Step 4: 去质心坐标
    # q_hat_i = q_i - q*,   p_hat_i = p_i - p*
    q_hat = target_pts[:, np.newaxis, :] - q_star[np.newaxis, :, :]  # (N, HW, 2)
    p_hat = source_pts[:, np.newaxis, :] - p_star[np.newaxis, :, :]  # (N, HW, 2)

    # Step 5: MLS 相似变换 (Similarity Transform) 逆向映射
    #
    # 论文公式: f_s(v) = |v - q*| / mu_s * sum_i { w_i * A_i^T } + p*
    #   其中 A_i = w_i * [ q_hat_i; -q_hat_i_perp ] * [ p_hat_i, p_hat_i_perp ]^{-1}
    #
    # 等价展开后:
    #   f_s(v) = (1/mu_s) * sum_i { w_i * [p_hat_i * (q_hat_i . d) + p_hat_i_perp * (q_hat_i_perp . d)] } + p*
    #   其中 d = v - q*,  perp(x,y) = (-y, x),  mu_s = sum_i { w_i * |q_hat_i|^2 }

    v_minus_qstar = v - q_star  # (HW, 2)

    # perp(x, y) = (-y, x)
    q_hat_perp = np.stack([-q_hat[..., 1], q_hat[..., 0]], axis=-1)  # (N, HW, 2)
    p_hat_perp = np.stack([-p_hat[..., 1], p_hat[..., 0]], axis=-1)  # (N, HW, 2)

    # mu_s = sum_i { w_i * |q_hat_i|^2 }
    mu = np.sum(weights * np.sum(q_hat ** 2, axis=-1), axis=0)  # (HW,)

    # 核心循环: 累加每个控制点的贡献
    result = np.zeros((num_pixels, 2), dtype=np.float64)

    for i in range(n):
        # dot_i  = q_hat_i . (v - q*)
        dot_i = np.sum(q_hat[i] * v_minus_qstar, axis=-1)  # (HW,)
        # cross_i = q_hat_i_perp . (v - q*)
        cross_i = np.sum(q_hat_perp[i] * v_minus_qstar, axis=-1)  # (HW,)

        # 加权累加: w_i * [ p_hat_i * dot_i + p_hat_i_perp * cross_i ]
        result += weights[i, :, np.newaxis] * (
            p_hat[i] * dot_i[:, np.newaxis] + p_hat_perp[i] * cross_i[:, np.newaxis]
        )

    # 归一化并加质心偏移: f_s(v) = result / mu + p*
    v_prime = result / (mu[:, np.newaxis] + eps) + p_star

    # Step 6: remap 双线性插值重采样
    v_prime[:, 0] = np.clip(v_prime[:, 0], 0, w - 1)
    v_prime[:, 1] = np.clip(v_prime[:, 1], 0, h - 1)

    map_x = v_prime[:, 0].reshape(h, w).astype(np.float32)
    map_y = v_prime[:, 1].reshape(h, w).astype(np.float32)

    warped_image = cv2.remap(img_array, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()