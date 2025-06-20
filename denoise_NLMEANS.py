import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 禁止用 Tkinter 图形界面

import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io.image import save_nifti


def denoise_NLMEANS(data, mask, affine, output_dir):

    sigma = estimate_sigma(data, N=32)

    # 执行NLMEANS降噪
    denoised_data = nlmeans(data, sigma=sigma, mask=mask, patch_radius=1, block_radius=2, rician=True)

    # 保存降噪结果
    denoised_path = output_dir / "denoised_data_processed.nii.gz"
    save_nifti(str(denoised_path), denoised_data, affine)

    # 生成降噪前后对比图
    axial_middle = data.shape[2] // 2
    before = data[:, :, axial_middle].T
    after = denoised_data[:, :, axial_middle].T
    difference = np.abs(after.astype(np.float64) - before.astype(np.float64))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(before, cmap="gray", origin="lower")
    ax[0].set_title("Before")
    ax[1].imshow(after, cmap="gray", origin="lower")
    ax[1].set_title("After")
    ax[2].imshow(difference, cmap="gray", origin="lower")
    ax[2].set_title("Difference")

    denoised_preview_path = output_dir / "denoised_preview.png"
    plt.savefig(str(denoised_preview_path), bbox_inches="tight")
    plt.close()


    return {
        "denoised_data": str(denoised_path),
        "denoised_preview": str(denoised_preview_path),
    }