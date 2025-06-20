import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import gaussian_laplace, binary_dilation
from dipy.io.image import save_nifti


def sharpen_laplace_dilation(data, mask, affine, output_dir, sharpening_factor=0.5, sigma=1.0):
    # 数据类型转换 (确保浮点运算)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    # 对掩膜进行膨胀操作
    dilated_mask = binary_dilation(mask, structure=np.ones((3, 3, 3)), iterations=1)

    # 高斯拉普拉斯滤波
    laplace = gaussian_laplace(data, sigma=sigma)

    # 锐化处理（原掩膜版本）
    sharpened_original_mask = data - sharpening_factor * laplace
    sharpened_original_mask = sharpened_original_mask * mask

    # 锐化处理（新掩膜版本）
    sharpened_new_mask = data - sharpening_factor * laplace
    sharpened_new_mask = sharpened_new_mask * dilated_mask

    # 保存结果
    sharpened_dilated_path = output_dir / "sharpened_data_dilated_processed.nii.gz"
    save_nifti(str(sharpened_dilated_path), sharpened_new_mask, affine)

    axial_middle = data.shape[2] // 2
    original_slice = data[:, :, axial_middle].T
    mask_slice = mask[:, :, axial_middle].T
    dilated_mask_slice = dilated_mask[:, :, axial_middle].T

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
    ax[0, 2].set_title("Dilated Mask Result")

    # 第二行：掩膜和差异对比
    ax[1, 0].imshow(mask_slice, cmap="gray", origin="lower")
    ax[1, 0].set_title("Original Mask")

    ax[1, 1].imshow(dilated_mask_slice, cmap="gray", origin="lower")
    ax[1, 1].set_title("Dilated Mask")

    diff = sharpened_new_mask[:, :, axial_middle].T - sharpened_original_mask[:, :, axial_middle].T
    vmax_diff = np.max(np.abs(diff))
    ax[1, 2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-vmax_diff, vmax=vmax_diff)
    ax[1, 2].set_title("Result Difference")

    plt.tight_layout()
    preview_path = output_dir / "sharpened_dilated_comparison.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "sharpened_dilated_data": str(sharpened_dilated_path),
        "comparison_dilated_preview": str(preview_path),
    }