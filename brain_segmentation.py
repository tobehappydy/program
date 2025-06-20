import matplotlib
matplotlib.use('Agg')  # 禁止 Tkinter 后端，适合服务器
import matplotlib.pyplot as plt
import numpy as np
from dipy.io.image import save_nifti
from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq


def brain_segmentation(data, affine, output_dir):
    fname = "segmentation"

    output_files = {}

    # 第一次分割（基础）
    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)

    # 保存分割后的图像数据
    segmented = data * mask

    segmented_path = output_dir / f"{fname}_data.nii.gz"
    save_nifti(str(segmented_path), segmented.astype(np.float32), affine)
    output_files['segmented_data'] = str(segmented_path)  # 前端会读取这个字段

    # 保存二值 mask 和原图 mask
    binary_mask_path = output_dir / f"{fname}_binary_mask.nii.gz"
    save_nifti(str(binary_mask_path), mask.astype(np.float32), affine)
    output_files['binary_mask'] = str(binary_mask_path)

    mask_path = output_dir / f"{fname}_mask.nii.gz"
    save_nifti(str(mask_path), b0_mask.astype(np.float32), affine)
    output_files['mask'] = str(mask_path)

    # 保存预览图（中间切片）
    sli = data.shape[2] // 2
    original = data[:, :, sli].T
    segmented_img = segmented[:, :, sli].T
    diff = segmented_img - original

    # 自动计算色标范围（避免全零差异图报错）
    vmax = max(np.abs(diff).max(), 1e-6)  # 使用1e-6作为最小值保护

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 改为3个子图

    # 原始图像
    ax[0].imshow(histeq(original.astype("float")), cmap="gray", origin="lower")
    ax[0].set_title("Original")
    ax[0].axis("off")

    # 分割后图像
    ax[1].imshow(histeq(segmented_img.astype("float")), cmap="gray", origin="lower")
    ax[1].set_title("Segmented")
    ax[1].axis("off")

    # 差异图
    ax[2].imshow(diff, cmap="coolwarm", origin="lower",
                 vmin=-vmax, vmax=vmax)
    ax[2].set_title("Difference")
    ax[2].axis("off")

    # 调整子图间距
    plt.tight_layout()

    # 添加整体标题
    plt.suptitle("Brain Segmentation Results", y=0.95, fontsize=16)

    preview_path = output_dir / f"{fname}_preview.png"
    plt.savefig(
        str(preview_path),
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white"
    )
    plt.close()
    output_files['segmented_preview'] = str(preview_path)

    # 第二阶段（带 autocrop）也保存
    b0_mask_crop, mask_crop = median_otsu(data, median_radius=4, numpass=4, autocrop=True)
    binary_mask_crop_path = output_dir / f"{fname}_binary_mask_crop.nii.gz"
    save_nifti(str(binary_mask_crop_path), mask_crop.astype(np.float32), affine)
    output_files['binary_mask_crop'] = str(binary_mask_crop_path)

    mask_crop_path = output_dir / f"{fname}_mask_crop_processed.nii.gz"
    save_nifti(str(mask_crop_path), b0_mask_crop.astype(np.float32), affine)
    output_files['mask_crop'] = str(mask_crop_path)

    return output_files
