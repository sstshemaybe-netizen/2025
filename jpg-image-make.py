import matplotlib
from sklearn.metrics import silhouette_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



# 数据：X 轴为 β 值（从 0.9 到 0），Y 轴为 NMI, ARI, ACC, F1，Z 轴为对应数值
x = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3])  # β 值，从 0.9 到 0
y_labels = np.array(['F1', 'ACC', 'NMI', 'ARI'])  # Y 轴标签
y_indices = np.arange(len(y_labels))      # 将 NMI, ARI, ACC, F1 映射为 0, 1, 2, 3

# Z 数据（对应四个指标的值，替换为你的实际数据）
# cora
# z = np.array([
#     [70.83, 63.24, 62.91, 64.34, 63.31, 69.19, 68.86, 64.07, 60.44, 60.34],     # NMI 数据
#     [73.71, 69.86, 69.46, 68.72, 69.05, 72.00, 71.56, 69.31, 58.54, 57.71],     # ARI 数据   # ACC 数据（示例）
#     [52.55, 49.44, 47.91, 46.75, 47.69, 48.66, 49.49, 49.37, 60.56, 59.34],
#     [54.36, 49.83, 49.85, 48.74, 48.77, 51.90, 50.45, 51.14, 62.50, 62.44]
# ])

# Z 数据（对应四个指标的值，替换为你的实际数据）
# corafull
# z = np.array([
#     [62.98, 59.88, 58.34, 58.79, 60.44, 60.44, 59.58, 59.57, 60.44, 60.34],     # NMI 数据
#     [63.24, 60.58, 61.48, 60.94, 60.12, 60.39, 58.43, 59.45, 58.54, 57.71],     # ARI 数据   # ACC 数据（示例）
#     [63.24, 61.81, 60.92, 60.44, 61.67, 60.26, 60.34, 61.23, 60.56, 59.34],
#     [63.04, 62.68, 62.55, 62.51, 62.51, 62.65, 62.65, 62.51, 62.50, 62.44]
# ])


# 创建 3D 图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 线条和标记点
for i, (label, z_data) in enumerate(zip(y_labels, z)):
    ax.plot(x, [y_indices[i]] * len(x), z_data, 'o-', label=label, linewidth=2, markersize=6)

# 设置标签、标题和图例
# ax.set_xlabel('β')
# ax.set_ylabel('Metric')
# ax.set_zlabel('Value')
# ax.set_title('The Ifluence of λ in Cora Dataset')
ax.set_yticks(y_indices)  # 设置 Y 轴刻度
ax.set_yticklabels(y_labels)  # 设置 Y 轴标签为 NMI, ARI, ACC, F1

# 反转 X 轴方向
ax.invert_xaxis()

# 添加图例
ax.legend()


plt.savefig('nmi_phy_lambda.png', dpi=300, bbox_inches='tight')