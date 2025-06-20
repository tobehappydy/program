import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from dipy.io.image import save_nifti
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.denoise.noise_estimate import estimate_sigma

def denoise_ASCM(data, mask, affine, output_dir):
    # 估计噪声水平 (根据设备调整N值)
    sigma = estimate_sigma(data, N=32)

    # 生成两种不同强度的降噪结果
    den_small = nlmeans(
        data, sigma=sigma, mask=mask,
        patch_radius=1, block_radius=1, rician=True
    )
    den_large = nlmeans(
        data, sigma=sigma, mask=mask,
        patch_radius=2, block_radius=1, rician=True
    )

    # 执行ASCM融合
    den_final = adaptive_soft_matching(
        data, den_small, den_large, sigma[0]
    )

    # 保存结果
    denoised_path = output_dir / "denoised_ascm_data_processed.nii.gz"
    save_nifti(str(denoised_path), den_final, affine)

    # 生成对比图
    axial_middle = data.shape[2] // 2
    before = data[:, :, axial_middle].T
    after = den_final[:, :, axial_middle].T
    difference = np.abs(after.astype(np.float64) - before.astype(np.float64))
    difference[~mask[:, :, axial_middle].T] = 0  # 掩膜外区域归零

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(before, cmap="gray", origin="lower")
    ax[0].set_title("Original")
    ax[1].imshow(after, cmap="gray", origin="lower")
    ax[1].set_title("ASCM Denoised")
    ax[2].imshow(difference, cmap="gray", origin="lower")
    ax[2].set_title("Difference")
    plt.tight_layout()

    preview_path = output_dir / "denoised_ascm_preview.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "denoised_ascm_data": str(denoised_path),
        "denoised_ascm_preview": str(preview_path),
    }