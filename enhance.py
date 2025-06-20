import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 禁止用 Tkinter 图形界面
from dipy.core.histeq import histeq
from dipy.io.image import save_nifti
import numpy as np


def enhance_histeq(data, affine, output_dir, num_bins=256):

    # 仅使用直方图均衡化
    enhanced_data = histeq(data.astype('float'), num_bins=num_bins)

    # 保存结果
    enhanced_path = output_dir / "histeq_enhanced_processed.nii.gz"
    save_nifti(str(enhanced_path), enhanced_data, affine)

    # 生成对比图
    axial_middle = data.shape[2] // 2
    original = data[:, :, axial_middle].T
    processed = enhanced_data[:, :, axial_middle].T
    diff = processed - original

    # 自动计算色标范围（避免全零差异图报错）
    vmax = max(np.abs(diff).max(), 1e-6)  # 使用1e-6作为最小值保护

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 改为3个子图

    # 原始图像
    ax[0].imshow(original, cmap="gray", origin="lower")
    ax[0].set_title("Original")

    # 增强后图像
    ax[1].imshow(processed, cmap="gray", origin="lower")
    ax[1].set_title("Enhanced")

    # 差异图（新增）
    ax[2].imshow(diff, cmap="coolwarm", origin="lower",
                 vmin=-vmax, vmax=vmax)
    ax[2].set_title("Difference")

    # 调整子图间距
    plt.tight_layout()

    preview_path = output_dir / "histeq_preview.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "enhanced_data": str(enhanced_path),
        "enhanced_preview": str(preview_path)
    }
