import matplotlib.pyplot as plt
import numpy as np

def save_as_image(tensor, filename, is_seg=False, n_classes=5):
    """
    tensor: (C, H, W) 或 (B, C, H, W)
    is_seg: 是否分割掩膜
    """
    # 创建保存目录
    os.makedirs("debug_images", exist_ok=True)

    # 处理批次维度
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # 取第一个样本

    tensor = tensor.detach().cpu().float()

    if is_seg:
        # 分割图处理（假设是one-hot格式）
        seg = tensor.argmax(0) if len(tensor.shape) == 3 else tensor.argmax(1)
        plt.imsave(filename, seg.numpy().astype(np.uint8), cmap='viridis')
    else:
        # 常规图像处理（假设是[-1,1]范围）
        img = tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
        plt.imsave(filename, img)
