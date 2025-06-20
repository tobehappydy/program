import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 禁止用 Tkinter 图形界面
import numpy as np
from pathlib import Path
from dipy.io.image import save_nifti, load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel, fractional_anisotropy, color_fa
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs


def DTI_reconstruction(dwi_data,  bvals, bvecs, affine, output_dir,
                median_radius=3, numpass=1, autocrop=True, dilate=2):

    # 1. 创建脑掩模
    maskdata, mask = median_otsu(
        dwi_data,
        vol_idx=range(10, 50),
        median_radius=median_radius,
        numpass=numpass,
        autocrop=autocrop,
        dilate=dilate
    )


    # 2. 构建DTI模型并拟合
    gtab = gradient_table(bvals, bvecs)
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)

    # 3. 计算各向异性指标
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)

    MD = tenfit.md
    RGB = color_fa(FA, tenfit.evecs)

    # 4. 保存结果文件
    fa_path = output_dir / "dti_fa.nii.gz"
    md_path = output_dir / "dti_md.nii.gz"
    rgb_path = output_dir / "dti_rgb.nii.gz"

    save_nifti(str(fa_path), FA.astype(np.float32), affine)
    save_nifti(str(md_path), MD.astype(np.float32), affine)
    save_nifti(str(rgb_path), np.array(255 * RGB, "uint8"), affine)

    # 5. 生成可视化切片
    axial_slice = FA.shape[2] // 2
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # FA图
    ax[0].imshow(FA[:, :, axial_slice].T, cmap="hot", origin="lower", vmin=0, vmax=1)
    ax[0].set_title("Fractional Anisotropy")

    # MD图
    ax[1].imshow(MD[:, :, axial_slice].T, cmap="gray", origin="lower")
    ax[1].set_title("Mean Diffusivity")



    plt.tight_layout()
    preview_path = output_dir / "dti_preview.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "fa_map": str(fa_path),
        "md_map": str(md_path),
        "rgb_map": str(rgb_path),
        "dti_preview": str(preview_path),
        "median_radius": median_radius,
        "numpass": numpass
    }

