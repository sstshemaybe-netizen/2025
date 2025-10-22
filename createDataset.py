import numpy as np

# 读取npz文件
from utils import generate_temporal_graph_dataset

dataset_name = 'Brain'
data = np.load("../data/" + dataset_name + ".npz")
print("Keys in .npz file:", data.files)
adjs = data["adjs"]
attmats = data["attmats"]
labels = data["labels"]


adjs, attmats, labels, node_active, category_introduce_time = generate_temporal_graph_dataset(
    num_time_steps=12,
    num_nodes=5000,
    total_categories=10,
    initial_categories=5,
    attr_dim=20
)

# 保存数据集
np.savez('temporal_graph_dataset.npz',
         adjs=adjs,
         attmats=attmats,
         labels=labels)
print("数据集已保存为 'temporal_graph_dataset.npz'")


# 输出关键信息
print("\n数据集关键信息:")
print(f"时间步数: {adjs.shape[0]}")
print(f"节点数量: {adjs.shape[1]}")
print(f"总类别数: {labels.shape[1]}")
print(f"类别引入时间: {sorted(category_introduce_time.items(), key=lambda x: x[1])}")

