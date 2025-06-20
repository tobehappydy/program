import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 禁止用 Tkinter 图形界面
from dipy.io.image import save_nifti, load_nifti
from dipy.segment.tissue import TissueClassifierHMRF
import numpy as np

from pathlib import Path

def tissue_segmentation(data, affine, output_dir, nclass=3, beta=0.1):
    # 执行组织分类
    if data.ndim == 4:
        data = data.mean(axis=-1)

    hmrf = TissueClassifierHMRF()
    initial_seg, final_seg, pve = hmrf.classify(data, nclass, beta)

    # 保存分类结果
    seg_path = output_dir / "tissue_segmentation_processed.nii.gz"
    save_nifti(str(seg_path), final_seg.astype(np.float32), affine)

    # 保存灰质概率图
    gm_path = output_dir / "gm_probability.nii.gz"
    save_nifti(str(gm_path), pve[..., 1], affine)  # 索引1对应GM

    # 保存白质概率图
    wm_path = output_dir / "wm_probability.nii.gz"
    save_nifti(str(wm_path), pve[..., 2], affine)  # 索引2对应WM

    # 生成可视化对比图
    axial_slice = data.shape[2] // 2
    fig = plt.figure(figsize=(15, 5))

    # 原始图像 (轴向)
    ax = fig.add_subplot(1, 4, 1)
    img_ax = np.rot90(data[..., axial_slice])
    ax.imshow(img_ax, cmap="gray")
    ax.axis("off")
    ax.set_title("Original (Axial)")

    # 分类结果 (轴向)
    ax = fig.add_subplot(1, 4, 2)
    seg_ax = np.rot90(final_seg[..., axial_slice])
    ax.imshow(seg_ax, vmin=0, vmax=nclass-1)
    ax.axis("off")
    ax.set_title("Segmentation")

    # 灰质概率图 (轴向)
    ax = fig.add_subplot(1, 4, 3)
    gm_ax = np.rot90(pve[:, :, axial_slice, 1])
    ax.imshow(gm_ax, cmap="hot")
    ax.axis("off")
    ax.set_title("GM Probability")

    # 白质概率图 (轴向)
    ax = fig.add_subplot(1, 4, 4)
    wm_ax = np.rot90(pve[:, :, axial_slice, 2])
    ax.imshow(wm_ax, cmap="hot")
    ax.axis("off")
    ax.set_title("WM Probability")



    plt.tight_layout()
    preview_path = output_dir / "segmentation_preview.png"
    plt.savefig(str(preview_path), bbox_inches="tight", dpi=150)
    plt.close()

    return {
        "segmentation_result": str(seg_path),
        "gm_probability": str(gm_path),
        "wm_probability": str(wm_path),
        "tissue_segmentation_preview": str(preview_path),
    }