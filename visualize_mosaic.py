import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_slice_mosaic(data, output_path, rows=6, cols=10, spacing=2):
    # 固定输出 6x8 布局，最多 48 张图
    total_slices = min(rows * cols, data.shape[2])
    step = max(1, data.shape[2] // total_slices)
    selected_slices = list(range(0, data.shape[2], step))[:total_slices]

    slice_height, slice_width = data.shape[0], data.shape[1]
    mosaic_height = rows * slice_height + (rows - 1) * spacing
    mosaic_width = cols * slice_width + (cols - 1) * spacing

    # 初始化马赛克图像为白色背景（可换成中性灰）
    mosaic = np.ones((mosaic_height, mosaic_width), dtype=np.float32)

    for idx, z in enumerate(selected_slices):
        row = idx // cols
        col = idx % cols
        y_start = row * (slice_height + spacing)
        x_start = col * (slice_width + spacing)
        slice_img = data[:, :, z].T.astype(np.float32)

        # 标准化每张切片
        slice_img -= slice_img.min()
        if slice_img.max() > 0:
            slice_img /= slice_img.max()

        mosaic[y_start:y_start + slice_height, x_start:x_start + slice_width] = slice_img

    # 动态调整 figsize，确保比例匹配
    dpi = 100
    fig_width = mosaic_width / dpi
    fig_height = mosaic_height / dpi

    plt.figure(figsize=(fig_width, fig_height), dpi=dpi, frameon=False)
    plt.imshow(mosaic, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()