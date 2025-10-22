import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def load_brain_dataset(file_path):
    """加载原始Brain数据集"""
    data = np.load(file_path)

    # 假设数据结构：adjs(时间步, 节点数, 节点数), attmats(节点数, 时间步, 属性维), labels(节点数, 类别数)
    adjs = data["adjs"]      # 形状: (12, N, N)
    attmats = data["attmats"]# 形状: (N, 12, D)
    labels = data["labels"]  # 形状: (N, C)

    print(f"原始数据加载完成:")
    print(f"时间步数: {adjs.shape[0]}, 节点数: {adjs.shape[1]}, 类别数: {labels.shape[1]}")

    # 确定每个节点的主类别
    main_categories = np.argmax(labels, axis=1)

    return adjs, attmats, labels, main_categories

def assign_node_timings(num_nodes, num_time_steps, main_categories, initial_categories=5):
    """
    为每个节点分配出现时间，确保类别动态增长
    新节点在出现时间前完全不存在于图中
    """
    total_categories = len(np.unique(main_categories))

    # 1. 确定每个类别的首次出现时间
    # 初始类别（前5类）在时间步0出现
    initial_cats = np.unique(main_categories)[:initial_categories]
    remaining_cats = [c for c in np.unique(main_categories) if c not in initial_cats]

    # 为类别分配出现时间
    cat_first_time = {cat: 0 for cat in initial_cats}
    if remaining_cats:
        # 剩余类别均匀分布在后续时间步
        step = max(1, (num_time_steps - 1) // len(remaining_cats))
        for i, cat in enumerate(remaining_cats):
            cat_first_time[cat] = min(1 + i * step, num_time_steps - 1)

    # 2. 为每个节点分配出现时间（必须晚于其类别的首次出现时间）
    node_start_time = np.zeros(num_nodes, dtype=int)
    for i in range(num_nodes):
        cat = main_categories[i]
        earliest_possible = cat_first_time[cat]
        # 节点出现时间在类别出现时间之后随机分配
        node_start_time[i] = np.random.randint(earliest_possible, num_time_steps)

    return node_start_time, cat_first_time

def modify_dataset(adjs, attmats, labels, node_start_time, num_time_steps):
    """
    修改数据集，确保新节点出现前无任何关联边
    在节点未出现的时间步，其对应的邻接矩阵行和列均为0
    """
    num_nodes = adjs.shape[1]
    modified_adjs = np.zeros_like(adjs)  # 初始化修改后的邻接矩阵

    for t in range(num_time_steps):
        # 当前时间步已出现的节点
        existing_nodes = np.where(node_start_time <= t)[0]
        num_existing = len(existing_nodes)

        if num_existing == 0:
            continue  # 无节点的时间步保持全0

        # 构建现有节点的索引映射（原始索引 -> 当前时间步的局部索引）
        node_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(existing_nodes)}

        # 提取现有节点之间的连接（从原始邻接矩阵）
        sub_adj = np.zeros((num_existing, num_existing), dtype=adjs.dtype)
        for i in range(num_existing):
            orig_i = existing_nodes[i]
            for j in range(i, num_existing):  # 无向图，只处理上三角
                orig_j = existing_nodes[j]
                sub_adj[i, j] = adjs[t, orig_i, orig_j]
                sub_adj[j, i] = sub_adj[i, j]  # 保持对称性

        # 将子邻接矩阵映射回完整矩阵（只保留现有节点间的连接）
        for i in range(num_existing):
            orig_i = existing_nodes[i]
            for j in range(num_existing):
                orig_j = existing_nodes[j]
                modified_adjs[t, orig_i, orig_j] = sub_adj[i, j]

    # 属性矩阵：未出现节点的属性设为0
    modified_attmats = attmats.copy()
    for i in range(num_nodes):
        # 节点未出现的时间步，属性设为0
        before_start = np.where(np.arange(num_time_steps) < node_start_time[i])[0]
        modified_attmats[i, before_start, :] = 0

    return modified_adjs, modified_attmats

def get_time_step_data(adjs, attmats, labels, main_categories, node_start_time, t):
    """
    提取指定时间步t的图数据
    返回：节点列表、边列表、节点属性、节点类别
    """
    num_nodes = adjs.shape[1]

    # 当前时间步存在的节点
    existing_nodes = np.where(node_start_time <= t)[0]
    if len(existing_nodes) == 0:
        return None, None, None, None

    # 提取边列表（只保留现有节点间的连接）
    edges = []
    adj_t = adjs[t]
    for i in existing_nodes:
        # 只找现有节点中与i相连的节点
        connected = [j for j in existing_nodes if j > i and adj_t[i, j] > 0]  # 避免重复边
        edges.extend([(i, j) for j in connected])

    # 提取节点属性
    node_attrs = {i: attmats[i, t] for i in existing_nodes}

    # 提取节点类别
    node_categories = {i: main_categories[i] for i in existing_nodes}

    return existing_nodes, edges, node_attrs, node_categories

def visualize_time_steps(adjs, node_start_time, main_categories, cat_first_time, num_time_steps=12):
    """可视化各时间步的图数据特征"""
    # 1. 每个时间步的节点数和边数
    node_counts = []
    edge_counts = []
    category_counts = []

    for t in range(num_time_steps):
        # 节点数
        nodes = np.where(node_start_time <= t)[0]
        node_counts.append(len(nodes))

        # 边数
        if len(nodes) < 2:
            edge_counts.append(0)
        else:
            adj_t = adjs[t]
            # 只计算现有节点间的边
            edges = np.sum(np.triu(adj_t[nodes][:, nodes], k=1))
            edge_counts.append(int(edges))

        # 类别数
        if len(nodes) == 0:
            category_counts.append(0)
        else:
            cats = np.unique(main_categories[nodes])
            category_counts.append(len(cats))

    # 绘制节点数和边数变化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(range(num_time_steps), node_counts, 'o-', color='b')
    axes[0].set_title('每个时间步的节点数量')
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('节点数')
    axes[0].set_xticks(range(num_time_steps))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(num_time_steps), edge_counts, 'o-', color='r')
    axes[1].set_title('每个时间步的边数量')
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('边数')
    axes[1].set_xticks(range(num_time_steps))
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(range(num_time_steps), category_counts, 'o-', color='g')
    axes[2].set_title('每个时间步的类别数量')
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('类别数')
    axes[2].set_xticks(range(num_time_steps))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 2. 可视化几个时间步的图结构（简化版）
    sample_steps = [0, 3, 6, 11]  # 选择几个有代表性的时间步
    fig, axes = plt.subplots(1, len(sample_steps), figsize=(5*len(sample_steps), 5))
    if len(sample_steps) == 1:
        axes = [axes]

    for i, t in enumerate(sample_steps):
        nodes, edges, _, node_cats = get_time_step_data(
            adjs, None, None, main_categories, node_start_time, t
        )

        if nodes is None or len(nodes) < 2:
            axes[i].set_title(f'时间步 {t} (无节点)')
            continue

        # 创建图并绘制（采样部分节点以提高可视化速度）
        sample_size = min(50, len(nodes))  # 最多显示50个节点
        sample_nodes = np.random.choice(nodes, sample_size, replace=False)
        sample_edges = [(u, v) for u, v in edges if u in sample_nodes and v in sample_nodes]

        G = nx.Graph()
        G.add_nodes_from(sample_nodes)
        G.add_edges_from(sample_edges)

        # 按类别着色
        colors = [node_cats[node] for node in sample_nodes]
        nx.draw(G, ax=axes[i], node_size=50, node_color=colors,
                cmap=plt.cm.tab10, with_labels=False)
        axes[i].set_title(f'时间步 {t} (节点数: {len(nodes)})')

    plt.tight_layout()
    plt.show()

def main():
    # 1. 加载原始数据
    brain_data_path = "../data/Brain.npz"  # 替换为你的数据集路径
    adjs, attmats, labels, main_categories = load_brain_dataset(brain_data_path)
    num_time_steps = adjs.shape[0]
    num_nodes = adjs.shape[1]

    # 2. 为节点分配出现时间（控制类别动态增长）
    node_start_time, cat_first_time = assign_node_timings(
        num_nodes=num_nodes,
        num_time_steps=num_time_steps,
        main_categories=main_categories,
        initial_categories=5  # 初始5个类别
    )

    # 3. 修改数据集，确保新节点出现前无关联边
    modified_adjs, modified_attmats = modify_dataset(
        adjs, attmats, labels, node_start_time, num_time_steps
    )

    # 4. 保存修改后的数据集
    output_path = "brain_dataset_time_aware.npz"
    np.savez(
        output_path,
        adjs=modified_adjs,
        attmats=modified_attmats,
        labels=labels,
        node_start_time=node_start_time,
        main_categories=main_categories
    )
    print(f"修改后的数据集已保存为: {output_path}")

    # 5. 示例：读取时间步t=0的数据
    t = 0
    nodes_t0, edges_t0, attrs_t0, cats_t0 = get_time_step_data(
        modified_adjs, modified_attmats, labels, main_categories, node_start_time, t
    )
    print(f"\n时间步 {t} 的图数据:")
    print(f"节点数: {len(nodes_t0) if nodes_t0 is not None else 0}")
    print(f"边数: {len(edges_t0) if edges_t0 is not None else 0}")
    print(f"类别数: {len(np.unique(list(cats_t0.values()))) if cats_t0 else 0}")

    # # 6. 可视化各时间步特征
    # visualize_time_steps(modified_adjs, node_start_time, main_categories, cat_first_time)

    # 输出类别出现时间
    print("\n类别首次出现时间:")
    for cat, time in sorted(cat_first_time.items(), key=lambda x: x[1]):
        print(f"类别 {cat}: 时间步 {time}")

if __name__ == "__main__":
    main()
