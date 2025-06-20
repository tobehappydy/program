# import matplotlib.pyplot as plt
# from dipy.core.histeq import histeq
# from dipy.io.image import save_nifti
# from scipy.ndimage import gaussian_laplace
#
#
# def enhance_histeq_sharpening(data, affine, output_dir, num_bins=256, sharpening_factor=0.3):
#
#     # 直方图均衡化
#     enhanced_data = histeq(data.astype('float'), num_bins=num_bins)
#
#     # 锐化处理 (使用拉普拉斯滤波器)
#     sharpened = enhanced_data - sharpening_factor * gaussian_laplace(enhanced_data, sigma=0.8)
#
#     # 保存结果
#     enhanced_path = output_dir / "enhanced_data_processed.nii.gz"
#     save_nifti(str(enhanced_path), sharpened, affine)
#
#     # 生成对比图
#     axial_middle = data.shape[2] // 2
#     original = data[:, :, axial_middle].T
#     processed = sharpened[:, :, axial_middle].T
#
#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].imshow(original, cmap="gray", origin="lower")
#     ax[0].set_title("Original")
#     ax[1].imshow(processed, cmap="gray", origin="lower")
#     ax[1].set_title("Enhanced")
#
#     preview_path = output_dir / "enhanced_preview.png"
#     plt.savefig(str(preview_path), bbox_inches="tight")
#     plt.close()
#
#     return {
#         "enhanced_data": str(enhanced_path),
#         "enhanced_preview": str(preview_path)
#     }


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import gaussian_laplace
from dipy.io.image import save_nifti



def sharpen_laplace(data, mask, affine, output_dir, sharpening_factor=0.5, sigma=1.0):

    # 数据类型转换 (确保浮点运算)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    # 高斯拉普拉斯滤波
    laplace = gaussian_laplace(data, sigma=sigma)

    # 锐化处理
    sharpened = data - sharpening_factor * laplace

    # 应用掩模并保存
    sharpened = sharpened * mask
    sharpened_path = output_dir / "sharpened_data_processed.nii.gz"
    save_nifti(str(sharpened_path), sharpened, affine)

    # 5. 生成对比图
    axial_middle = data.shape[2] // 2
    original = data[:, :, axial_middle].T
    processed = sharpened[:, :, axial_middle].T

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original, cmap="gray", origin="lower",
                 vmin=np.percentile(original, 5),
                 vmax=np.percentile(original, 95))
    ax[0].set_title("Original")

    ax[1].imshow(processed, cmap="gray", origin="lower",
                 vmin=np.percentile(original, 5),
                 vmax=np.percentile(original, 95))
    ax[1].set_title("Sharpened")

    diff = processed - original
    vmax = np.max(np.abs([diff.min(), diff.max()]))  # 自动计算色标范围
    ax[2].imshow(diff, cmap="coolwarm", origin="lower",
                 vmin=-vmax, vmax=vmax)
    ax[2].set_title("Difference")

    preview_path = output_dir / "sharpened_preview.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "sharpened_data": str(sharpened_path),
        "sharpened_preview": str(preview_path)
    }