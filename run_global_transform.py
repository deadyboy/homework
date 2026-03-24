import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 1. 旋转和缩放矩阵 (围绕图像中心)
    # cv2.getRotationMatrix2D 返回 2x3 矩阵，使用预定义的 to_3x3 转换为 3x3 齐次矩阵
    M_rot_scale_2x3 = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    M_rot_scale = to_3x3(M_rot_scale_2x3)

    # 2. 翻转矩阵 (水平翻转)
    if flip_horizontal:
        # 围绕图像中心 x = cx 进行对称翻转：x' = -x + 2*cx
        M_flip = np.array([
            [-1, 0, 2 * cx],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], dtype=np.float32)
    else:
        M_flip = np.eye(3, dtype=np.float32)

    # 3. 平移矩阵
    M_translate = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # 4. 矩阵组合运算
    # 按照执行顺序的逆序进行左乘：平移 <- 旋转/缩放 <- 翻转
    M_composite = M_translate @ M_rot_scale @ M_flip

    # 提取最终的 2x3 仿射变换矩阵以供 OpenCV 使用
    M_affine = M_composite[:2, :]

    # 5. 应用复合仿射变换 (保持背景填充颜色与 pad_size 逻辑中的白色一致)
    transformed_image = cv2.warpAffine(image, M_affine, (w, h), borderValue=(255, 255, 255))

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch(server_name="0.0.0.0", server_port=7861, share=False)