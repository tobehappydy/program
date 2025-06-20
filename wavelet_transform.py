import numpy as np
import pywt
import matplotlib.pyplot as plt
from dipy.io.image import save_nifti
from pathlib import Path


def wavelet_transform(data, affine, output_dir, wavelet="haar", level=1):
    # 应用小波变换
    coeffs = pywt.wavedec(data, wavelet, mode="periodization", level=level)

    # 仅使用低频分量 (Approximation) 进行重构
    reconstructed_data = pywt.waverec(coeffs, wavelet, mode="periodization")

    # 保持数据尺寸一致
    reconstructed_data = reconstructed_data[:data.shape[0], :data.shape[1], :data.shape[2]]

    # 定义输出文件名
    wavelet_filename = "wavelet_transformed.nii.gz"
    preview_filename = "wavelet_preview.png"

    # 保存结果
    wavelet_path = output_dir / wavelet_filename
    save_nifti(str(wavelet_path), reconstructed_data.astype(np.float32), affine)

    # 生成对比图
    axial_middle = data.shape[2] // 2
    original = data[:, :, axial_middle].T
    processed = reconstructed_data[:, :, axial_middle].T

    # 创建三视图对比
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    ax[0].imshow(original, cmap="gray", origin="lower")
    ax[0].set_title("Original")

    # 小波变换后图像
    ax[1].imshow(processed, cmap="gray", origin="lower")
    ax[1].set_title(f"Wavelet Transformed ({wavelet}, Level {level})")

    # 差异图
    diff = processed - original
    vmax = max(np.abs(diff).max(), 1e-6)
    ax[2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-vmax, vmax=vmax)
    ax[2].set_title("Difference")

    plt.tight_layout()

    preview_path = output_dir / "wavelet_transformed.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    # 返回只包含文件名的字典
    return {
        "wavelet_transformed": str(wavelet_path),
        "wavelet_preview": str(preview_path)
    }
