import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from dipy.io.image import save_nifti
from pathlib import Path


def smooth_gaussian(data, affine, output_dir, sigma=1.0):

    # 应用高斯滤波
    smoothed_data = gaussian_filter(data, sigma=sigma)

    # 定义输出文件名
    smoothed_filename = "smoothed_processed.nii.gz"
    preview_filename = "smooth_preview.png"

    # 保存结果
    smoothed_path = output_dir / smoothed_filename
    save_nifti(str(smoothed_path), smoothed_data, affine)

    # 生成对比图
    axial_middle = data.shape[2] // 2
    original = data[:, :, axial_middle].T
    processed = smoothed_data[:, :, axial_middle].T

    # 创建三视图对比
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    ax[0].imshow(original, cmap="gray", origin="lower")
    ax[0].set_title("Original")

    # 平滑后图像
    ax[1].imshow(processed, cmap="gray", origin="lower")
    ax[1].set_title(f"Smoothed (σ={sigma})")

    # 差异图
    diff = processed - original
    vmax = max(np.abs(diff).max(), 1e-6)
    ax[2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-vmax, vmax=vmax)
    ax[2].set_title("Difference")

    plt.tight_layout()

    preview_path = output_dir / "smoothed.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    # 返回只包含文件名的字典
    return {
        "smoothed_data": str(smoothed_path),
        "smoothed_preview": str(preview_path)
    }