from dipy.workflows.io import FetchFlow
import os
from tqdm import tqdm  # 用于显示进度条

# 用作 DIPY 数据集下载的函数
def fetch_dipy_data(dataset_names, output_dir):

    # 创建输出目录(如果不存在)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化FetchFlow
    fetch_flow = FetchFlow()

    results = {}

    # 检查可用数据集
    available_data = FetchFlow.get_fetcher_datanames().keys()
    print(f"可用数据集: {list(available_data)}")

    # 遍历要下载的数据集
    for dataset in tqdm(dataset_names, desc="下载进度"):
        if dataset not in available_data:
            print(f"警告: 数据集 '{dataset}' 不可用，跳过")
            results[dataset] = "不可用"
            continue

        try:
            # 运行下载
            print(f"\n正在下载数据集: {dataset}")
            fetch_flow.run([dataset], out_dir=output_dir)
            results[dataset] = "成功"
            print(f"数据集 {dataset} 已保存到: {output_dir}")
        except Exception as e:
            print(f"下载 {dataset} 时出错: {str(e)}")
            results[dataset] = f"失败: {str(e)}"

    return results


if __name__ == "__main__":
    # 配置参数
    datasets_to_fetch = ['	sherbrooke_3shell','stanford_t1','qtdMRI_test_retest_2subjects','fetch_ivim','bundle_atlas_hcp842','cenir_multib','syn_data','taiwan_ntu_dsi']  # 要下载的数据集列表
    output_directory = 'D:/dataset/dipy_data'  # 保存目录

    # 执行下载
    print("开始下载DIPY数据集...")
    download_results = fetch_dipy_data(datasets_to_fetch, output_directory)

    # 打印结果摘要
    print("\n下载结果摘要:")
    for dataset, status in download_results.items():
        print(f"{dataset}: {status}")