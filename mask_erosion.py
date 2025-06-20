import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import gaussian_laplace, binary_erosion
from dipy.io.image import save_nifti


def sharpen_laplace_erosion(data, mask, affine, output_dir, sharpening_factor=0.5, sigma=1.0):
    # 数据类型转换 (确保浮点运算)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    # 对掩膜进行腐蚀操作
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3, 3)), iterations=2)

    # 高斯拉普拉斯滤波
    laplace = gaussian_laplace(data, sigma=sigma)

    # 锐化处理（原掩膜版本）
    sharpened_original_mask = data - sharpening_factor * laplace
    sharpened_original_mask = sharpened_original_mask * mask

    # 锐化处理（新掩膜版本）
    sharpened_new_mask = data - sharpening_factor * laplace
    sharpened_new_mask = sharpened_new_mask * eroded_mask

    # 保存结果
    sharpened_eroded_path = output_dir / "sharpened_data_eroded_processed.nii.gz"
    save_nifti(str(sharpened_eroded_path), sharpened_new_mask, affine)

    # 对比图：展示三种状态
    axial_middle = data.shape[2] // 2
    original_slice = data[:, :, axial_middle].T
    mask_slice = mask[:, :, axial_middle].T
    eroded_mask_slice = eroded_mask[:, :, axial_middle].T

    # 计算显示范围
    vmin = np.percentile(original_slice, 5)
    vmax = np.percentile(original_slice, 95)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：图像对比
    ax[0, 0].imshow(original_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("Original Image")

    ax[0, 1].imshow(sharpened_original_mask[:, :, axial_middle].T,
                    cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("Original Mask Result")

    ax[0, 2].imshow(sharpened_new_mask[:, :, axial_middle].T,
                    cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax[0, 2].set_title("Eroded Mask Result")

    # 第二行：掩膜和差异对比
    ax[1, 0].imshow(mask_slice, cmap="gray", origin="lower")
    ax[1, 0].set_title("Original Mask")

    ax[1, 1].imshow(eroded_mask_slice, cmap="gray", origin="lower")
    ax[1, 1].set_title("Eroded Mask")

    diff = sharpened_new_mask[:, :, axial_middle].T - sharpened_original_mask[:, :, axial_middle].T
    vmax_diff = np.max(np.abs(diff))
    ax[1, 2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-vmax_diff, vmax=vmax_diff)
    ax[1, 2].set_title("Result Difference")

    plt.tight_layout()
    preview_path = output_dir / "sharpened_eroded_comparison.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "sharpened_eroded_data": str(sharpened_eroded_path),
        "comparison_eroded_preview": str(preview_path),
    }