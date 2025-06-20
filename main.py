import numpy as np
from pathlib import Path
from dipy.io.image import load_nifti
from datetime import datetime
import nibabel
import matplotlib.pyplot as plt


def get_input_file():
    user_input = input("请输入输入NIfTI文件路径 (按回车退出): ").strip()
    if user_input:
        return user_input
    else:
        print("程序已退出。")
        exit()



def create_output_dir():
    project_root = Path(__file__).parent  # 获取项目根目录
    result_dir = project_root / "result"
    result_dir.mkdir(exist_ok=True)  # 创建result目录(如果不存在)

    # 创建带时间戳的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = result_dir / timestamp
    output_dir.mkdir(exist_ok=True)
    return output_dir

def visualize_with_orthoslicer(file_path):

    img = nibabel.load(file_path)
    data = img.get_fdata()
    nibabel.viewers.OrthoSlicer3D(data).show()


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

def main():
    print("=== 医学图像处理程序 ===")

    while True:  # 主循环
        # 1. 文件输入阶段
        filepath = get_input_file()
        if filepath is None:  # 用户选择退出
            print("程序已退出。")
            break

        try:
            # 加载数据
            data, affine = load_nifti(filepath)
            data = np.squeeze(data)

            # 2. 功能选择阶段
            while True:  # 功能选择循环
                print("\n请选择要执行的功能:")
                print("1. 脑部分割")
                print("2. 降噪")
                print("3. 增强（直方图均衡化）")
                print("4. 锐化增强")
                print("r. 重新选择文件")
                print("q. 退出程序")
                choice = input("请输入选择: ").strip().lower()

                if choice == 'q':
                    print("程序已退出。")
                    return  # 完全退出程序

                if choice == 'r':
                    break  # 跳出功能选择循环，重新选择文件

                # 创建输出目录
                output_dir = create_output_dir()

                # 3. 执行处理功能
                if choice == '1':
                    # 这里调用brain_segmentation函数
                    from brain_segmentation import brain_segmentation
                    output_files = brain_segmentation(data, affine, output_dir)
                    print(f"\n[脑分割完成] 结果已保存到: {output_dir}")
                elif choice == '2':
                    # 这里调用denoise_NLMEANS函数
                    from denoise_NLMEANS import denoise_NLMEANS
                    mask = np.ones(data.shape, dtype=bool)
                    output_files = denoise_NLMEANS(data, mask, affine, output_dir)
                    print(f"\n[降噪完成] 结果已保存到: {output_dir}")
                elif choice == '3':
                    # 这里调用enhance_histeq函数
                    from enhance import enhance_histeq
                    output_files = enhance_histeq(data, affine, output_dir, num_bins=256)
                    print(f"\n[增强完成] 结果已保存到: {output_dir}")
                elif choice == '4':
                    from enhance_with_sharpening import sharpen_laplace
                    mask = np.ones(data.shape, dtype=bool)  # 创建一个全1的掩模
                    output_files = sharpen_laplace(data, mask, affine, output_dir, sharpening_factor=0.5, sigma=1.0)
                    print(f"\n[锐化增强完成] 结果已保存到: {output_dir}")

                else:
                    print("无效的选择，请重新输入")
                    continue

                # 4. 可视化选择
                restart_main_loop = False
                while True:  # 可视化选择循环
                    viz_choice = input("\n是否进一步可视化结果？(yes/no): ").strip().lower()

                    if viz_choice == 'yes':
                        processed_files = list(output_dir.glob("*processed.nii.gz"))
                        if not processed_files:
                            print("未找到任何后缀为 processed.nii.gz 的文件进行可视化。")
                        else:
                            for processed_path in processed_files:
                                print(f"正在可视化文件: {processed_path}")
#                                visualize_with_orthoslicer(str(processed_path))
                                generate_slice_mosaic(data, output_dir / "slice_mosaic.png")
                                print(f"切片马赛克已保存到: {output_dir / 'slice_mosaic.png'}")
                        break
                    elif viz_choice == 'no':
                        return  # 完全退出程序
                    else:
                        print("无效输入，请输入 yes/no")

                if restart_main_loop:
                    break

        except Exception as e:
            print(f"\n处理过程中发生错误: {str(e)}")
            print("请检查输入文件或参数后重试")
            continue


if __name__ == "__main__":
    main()