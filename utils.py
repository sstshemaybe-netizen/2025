import copy
from time import time
import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score, accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import scipy.sparse as sp
from torch import nn
from torch_geometric.data import Data
from torch_geometric.graphgym import optim
from torch_geometric.nn import GCNConv

from kmeans_gpu_3 import kmeans


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_graph_data(dataset_name, show_details=False, seed=0):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    load_path = "../data/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")

    # # # 我加的代码 用于生成加噪数据
    # feat = generated(dataset_name, 0.01, 100, seed=seed)
    feat = sp.csr_matrix(feat)
    return feat, label, adj

def load_npz(dataset_name):
    data = np.load("../data/" + dataset_name + ".npz")
    print("Keys in .npz file:", data.files)

    adj_data = data["adj_data"]
    adj_indices = data["adj_indices"]
    adj_indptr = data["adj_indptr"]
    adj_shape = tuple(data["adj_shape"])  # 转换为元组（如(13752, 13752)）

    # 构建稀疏邻接矩阵
    adj_matrix = csr_matrix(
        (adj_data, adj_indices, adj_indptr),
        shape=adj_shape
    )

    # 提取节点特征和标签
    feature_matrix = csr_matrix(
        (data['attr_data'], data['attr_indices'], data['attr_indptr']),
        shape=tuple(data['attr_shape'])
    )
    labels = data['labels']
    return feature_matrix, labels, adj_matrix

def load_npz_time(dataset_name):
    data = np.load("../data/" + dataset_name + ".npz")
    print("Keys in .npz file:", data.files)

    adjs = data["adjs"]
    attmats = data["attmats"]
    labels = data["labels"]
    print("adjs:", adjs.shape)
    print("attmats:", attmats.shape)
    print("labels:", labels.shape)


# 训练模型
def train(model, optimizer, x, edge_index, y):
    model.train()
    optimizer.zero_grad()
    print("edge_index:", edge_index.shape)
    out = model(x, edge_index)
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def preprocess_features(features):
    """
    row-normalize node attributes
    args:
        features: input node attributes
    returns:
        normalized node attributes
    """
    rowsum = np.array(features.sum(1))
    rowsum[rowsum==0] = -np.inf
    r_inv = np.power(rowsum, -1).flatten()
    r_mat_inv = sp.diags(r_inv)
    norm_features = r_mat_inv.dot(features).todense()
    return norm_features


# 测试模型
def test(model, x, edge_index, y):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        print("pred:", pred)

def preprocess_graph(adj, layer, norm='sym', renorm=True, k=1):
    adj = sp.coo_matrix(adj)
    # 构造单位矩阵
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - k*adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'right':
        degree_mat_inv = sp.diags(np.power(rowsum, -1).flatten())
        adj_normalized = adj_.dot(degree_mat_inv).tocoo()
        laplacian = ident - adj_normalized

    elif norm == 'global':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, 1).flatten())
        laplacian = degree_mat_inv_sqrt - adj_
        laplacian = laplacian.tocoo()
    elif norm == 'high':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = adj_normalized
    elif norm == 'all':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

        laplacian_2 = ident - k*(adj_normalized)

        laplacian = laplacian_2

    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        # print("reg[i]:", reg[i])
        adjs.append(ident - (reg[i] * laplacian))
    # print("adjs:", adjs)
    return adjs

def clustering_3(feature, true_labels, cluster_num):
    predict_labels, centers, dis = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda:0")

    nmi, ari = eva(true_labels, predict_labels.numpy(), show_details=False)
    return 100 * nmi, 100 * ari, predict_labels.numpy(), centers, dis

def clustering(feature, true_labels, cluster_num):
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda:0")
    nmi, ari = eva(true_labels, predict_labels.numpy(), show_details=False)
    return round(100 * nmi, 2), round(100 * ari, 2)

def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # ss = silhouette_score(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))

    return nmi, ari

def kl_divergence(P, Q):
    return (P * (torch.log(P) - torch.log(Q))).sum()

def convert_scipy_torch_sp(sp_adj):
    sp_adj = sp_adj.tocoo()
    indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
    sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
    return sp_adj


def aug_feature_dropout(input_feat, drop_rate=0.2):
    """
    dropout features for augmentation.
    args:
        input_feat: input features
        drop_rate: dropout rate
    returns:
        aug_input_feat: augmented features
    """
    aug_input_feat = input_feat.squeeze(0)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_rate)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    aug_input_feat = aug_input_feat.unsqueeze(0)

    return aug_input_feat

def aug_feature_shuffle(input_feat):
    """
    shuffle the features for fake samples.
    args:
        input_feat: input features
    returns:
        aug_input_feat: augmented features
    """
    fake_input_feat = input_feat[:, np.random.permutation(input_feat.shape[1]), :]
    return fake_input_feat

def remove_outliers(data, z_threshold=6):
    """基于Z-score的双维度离群点过滤"""
    z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
    return np.all(z_scores < z_threshold, axis=1)

def initialize_dataset(features, A, true_labels, num_classes=5, sample_ratio=0.5):
    """初始化数据集，选择指定数量的类别并各取一定比例的样本"""
    # 设置随机种子
    random.seed(64)
    np.random.seed(64)

    # 获取所有唯一类别并选择指定数量
    unique_labels = np.unique(true_labels)
    if len(unique_labels) < num_classes:
        raise ValueError(f"可用类别少于{num_classes}个，无法完成选择")

    selected_labels = random.sample(list(unique_labels), num_classes)
    print(f"选中的{num_classes}个类别: {selected_labels}")

    # 为每个选中的类别选择指定比例的数据
    selected_indices = []
    for label in selected_labels:
        label_indices = np.where(true_labels == label)[0]
        select_count = int(len(label_indices) * sample_ratio)
        selected_for_label = np.random.choice(label_indices, select_count, replace=False)
        selected_indices.extend(selected_for_label)

    # 确保索引唯一并排序
    selected_indices = np.unique(selected_indices)
    selected_indices.sort()

    # 计算剩余索引
    all_indices = np.arange(len(true_labels))
    remaining_indices = np.setdiff1d(all_indices, selected_indices)

    # 筛选初始数据
    filtered_features = features[selected_indices]
    filtered_labels = true_labels[selected_indices]

    # 筛选邻接矩阵
    mask = np.zeros(A.shape[0], dtype=bool)
    mask[selected_indices] = True
    filtered_A = A[mask][:, mask]

    print(f"初始选择了 {len(selected_indices)} 个样本")
    print(f"剩余 {len(remaining_indices)} 个样本")

    return filtered_features, filtered_A, filtered_labels, selected_indices, remaining_indices, selected_labels

def add_incremental_data(features, A, old_labels, true_labels, current_indices, remaining_indices, selected_labels, batch_size=500):
    """
    从所有剩余样本中添加样本，每次最多添加batch_size个（默认500），不足时全部加入，并返回新加入的节点索引。

    参数:
        features: 原始特征矩阵
        A: 原始邻接矩阵
        true_labels: 原始标签
        current_indices: 当前已选择的样本索引
        remaining_indices: 所有剩余样本的索引
        selected_labels: 已选择的标签（未在函数中使用，但保留参数以兼容原接口）
        batch_size: 每次添加的最大样本数，默认为500

    返回:
        new_features: 添加后的特征矩阵
        new_A: 添加后的邻接矩阵
        new_labels: 添加后的标签
        new_current_indices: 更新后的已选样本索引
        new_remaining_indices: 更新后的剩余样本索引
        added_count: 本次实际添加的样本数
        add_indices: 本次添加的样本索引
    """
    # 设置随机种子确保可复现性
    random.seed(64)
    np.random.seed(64)

    # 检查是否有剩余样本
    if len(remaining_indices) == 0:
        print("没有可添加的剩余样本")
        return None, None, None, current_indices, remaining_indices, 0, np.array([])


    old_labels = old_labels
    print("old_labels:", old_labels.shape)
    old_labels = torch.tensor(old_labels, dtype=torch.int)
    # 确定本次添加的样本数量（最多500个）
    add_count = min(batch_size, len(remaining_indices))

    # 从所有剩余样本中随机选择
    add_indices = np.random.choice(remaining_indices, add_count, replace=False)

    # 更新当前样本索引和剩余样本索引
    new_current_indices = np.append(current_indices, add_indices)

    new_remaining_indices = np.setdiff1d(remaining_indices, add_indices)

    # 筛选添加后的特征矩阵
    new_features = features[new_current_indices]

    # 筛选添加后的标签
    new_labels = torch.full((old_labels.shape[0]+add_count,), -1, dtype=torch.int)
    new_labels[:old_labels.shape[0]] = old_labels
    # 筛选添加后的邻接矩阵
    mask = np.zeros(A.shape[0], dtype=bool)
    mask[new_current_indices] = True
    new_A = A[mask][:, mask]

    # 打印添加信息
    print(f"添加了 {add_count} 个样本，当前总样本数: {len(new_current_indices)}")
    print(f"剩余样本数: {len(new_remaining_indices)}")
    print(f"本次添加的节点索引: {add_indices.tolist()}")

    return new_features, new_A, new_labels, new_current_indices, new_remaining_indices, add_count, add_indices


def corrupt_labels_within_range(original_labels, error_ratio=0.2, random_seed=64):
    """
    将一部分标签替换为错误标签，但错误标签仍在原始标签的类别范围内

    参数:
        original_labels: 原始标签数组
        error_ratio: 错误标签的比例，默认为0.2（20%）
        random_seed: 随机种子，保证结果可复现

    返回:
        corrupted_labels: 包含错误标签的新标签数组
        error_mask: 标记哪些位置是错误标签的布尔数组
    """
    # 设置随机种子
    np.random.seed(random_seed)

    # 复制原始标签作为基础
    corrupted_labels = np.copy(original_labels)

    # 获取原始标签的所有唯一类别
    unique_classes = np.unique(original_labels)
    num_classes = len(unique_classes)

    # 如果只有一个类别，无法生成错误标签
    if num_classes <= 1:
        raise ValueError("原始标签至少需要包含2个不同类别才能生成错误标签")

    # 确定需要变为错误标签的样本数量
    total_samples = len(original_labels)
    error_count = int(total_samples * error_ratio)

    # 随机选择要变为错误标签的样本索引
    error_indices = np.random.choice(total_samples, error_count, replace=False)

    # 为每个选中的样本生成错误标签（不同于原始标签）
    for idx in error_indices:
        original_class = original_labels[idx]

        # 从除原始类别外的其他类别中随机选择一个作为错误标签
        possible_wrong_classes = unique_classes[unique_classes != original_class]
        wrong_class = np.random.choice(possible_wrong_classes)

        # 设置错误标签
        corrupted_labels[idx] = wrong_class

    # 创建错误标签的掩码（True表示该位置是错误标签）
    error_mask = np.zeros(total_samples, dtype=bool)
    error_mask[error_indices] = True

    # 验证错误比例
    actual_error_ratio = np.mean(corrupted_labels != original_labels)
    print(f"实际错误标签比例: {actual_error_ratio:.2%}")

    return corrupted_labels, error_mask



def preprocess_graph(adj, layer, norm='sym', renorm=True, k=1):
    adj = sp.coo_matrix(adj)
    # 构造单位矩阵
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - k*adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'right':
        degree_mat_inv = sp.diags(np.power(rowsum, -1).flatten())
        adj_normalized = adj_.dot(degree_mat_inv).tocoo()
        laplacian = ident - adj_normalized

    elif norm == 'global':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, 1).flatten())
        laplacian = degree_mat_inv_sqrt - adj_
        laplacian = laplacian.tocoo()
    elif norm == 'high':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = adj_normalized
    elif norm == 'all':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

        laplacian_2 = ident - k*(adj_normalized)

        laplacian = laplacian_2

    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        # print("reg[i]:", reg[i])
        adjs.append(ident - (reg[i] * laplacian))
    # print("adjs:", adjs)
    return adjs


class SequentialGraphReader:
    """按时间序列依次读取图数据的工具类（包含自环边）"""

    def __init__(self, data_path: str):
        """初始化读取器，加载数据集"""
        self.data = self._load_data(data_path)
        self.num_time_steps = self.data["adjs"].shape[0]
        self.current_step = 0  # 当前读取的时间步索引

        # 打印数据集基本信息
        print(f"已加载动态图数据集，包含 {self.num_time_steps} 个时间步")
        print(f"总节点数: {self.data['adjs'].shape[1]}, 总类别数: {self.data['labels'].shape[1]}")

    def _load_data(self, data_path: str) -> Dict:
        """加载修改后的动态图数据集"""
        loaded = np.load(data_path)
        return {
            "adjs": loaded["adjs"],                  # (T, N, N) 邻接矩阵
            "attmats": loaded["attmats"],            # (N, T, D) 节点属性
            "labels": loaded["labels"],              # (N, C) 节点标签
            "node_start_time": loaded["node_start_time"],  # (N,) 节点出现时间
            "main_categories": loaded["main_categories"]   # (N,) 节点主类别
        }

    def get_next_graph(self) -> Dict:
        """读取下一个时间步的图数据，返回当前时间步的完整图信息"""
        if self.current_step >= self.num_time_steps:
            print("已读取完所有时间步的数据")
            return None

        # 获取当前时间步的图数据
        t = self.current_step
        graph_data = self.get_graph_at_step(t)

        # 移动到下一个时间步
        self.current_step += 1
        return graph_data

    def get_graph_at_step(self, t: int) -> Dict:
        """获取指定时间步t的图数据（包含自环边）"""
        if t < 0 or t >= self.num_time_steps:
            raise ValueError(f"时间步t必须在0到{self.num_time_steps-1}之间")

        # 筛选当前时间步已出现的节点
        existing_mask = self.data["node_start_time"] <= t
        existing_nodes = np.where(existing_mask)[0]
        num_nodes = len(existing_nodes)

        # 提取邻接矩阵（仅包含现有节点）
        adj_matrix = self.data["adjs"][t][existing_mask][:, existing_mask]

        # 提取边列表（u, v），使用原始节点ID
        edges = []
        # 1. 先添加原有边（节点间的连接）
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # 无向图，避免重复边
                if adj_matrix[i, j] > 0:
                    edges.append((existing_nodes[i], existing_nodes[j]))

        # 2. 为每个存在的节点添加自环边 (node_id, node_id)
        for node_id in existing_nodes:
            edges.append((node_id, node_id))  # 自环边：节点自身到自身的连接

        # 提取节点属性（当前时间步的属性）
        node_attrs = self.data["attmats"][existing_mask, t, :]

        # 提取节点类别信息
        node_categories = self.data["main_categories"][existing_mask]
        category_set = np.unique(node_categories)

        return {
            "time_step": t,
            "node_ids": existing_nodes,          # 原始节点ID列表
            "num_nodes": num_nodes,              # 当前节点数量
            "edges": edges,                      # 边列表（包含自环）
            "adj_matrix": adj_matrix,            # 邻接矩阵（原始，不含自环）
            "adj_matrix_with_loop": self._add_self_loop(adj_matrix),  # 含自环的邻接矩阵
            "node_attrs": node_attrs,            # 节点属性矩阵
            "node_categories": node_categories,  # 节点类别列表
            "category_set": category_set,        # 存在的类别集合
            "num_categories": len(category_set)  # 当前类别数量
        }

    def _add_self_loop(self, adj_matrix):
        """为邻接矩阵添加自环（对角线元素设为1）"""
        # 复制原始矩阵，避免修改原数据
        adj_with_loop = adj_matrix.copy()
        # 对角线元素设为1（自环）
        np.fill_diagonal(adj_with_loop, 1)
        return adj_with_loop

    def reset(self):
        """重置读取指针，从头开始读取"""
        self.current_step = 0
        print("已重置读取指针，将从时间步0开始读取")

    def read_all_sequentially(self) -> List[Dict]:
        """按顺序读取所有时间步的图数据，返回列表"""
        self.reset()
        all_graphs = []
        print(f"开始按顺序读取所有 {self.num_time_steps} 个时间步的图数据...")

        for t in range(self.num_time_steps):
            graph = self.get_next_graph()
            if graph:
                all_graphs.append(graph)
                # 打印进度
                if (t + 1) % 3 == 0:
                    print(f"已读取时间步 {t}，节点数: {graph['num_nodes']}, "
                          f"边数（含自环）: {len(graph['edges'])}")

        print("所有时间步数据读取完成")
        return all_graphs

def analyze_sequential_data(all_graphs: List[Dict]):
    """分析按顺序读取的时序图数据，生成统计信息和可视化"""
    # 提取时序特征
    time_steps = [g["time_step"] for g in all_graphs]
    node_counts = [g["num_nodes"] for g in all_graphs]
    edge_counts = [len(g["edges"]) for g in all_graphs]
    category_counts = [g["num_categories"] for g in all_graphs]

    # 绘制时序趋势
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 节点数量趋势
    axes[0].plot(time_steps, node_counts, 'o-', color='blue')
    axes[0].set_title('节点数量随时间变化')
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('节点数')
    axes[0].set_xticks(time_steps)
    axes[0].grid(alpha=0.3)

    # 边数量趋势
    axes[1].plot(time_steps, edge_counts, 's-', color='red')
    axes[1].set_title('边数量随时间变化')
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('边数')
    axes[1].set_xticks(time_steps)
    axes[1].grid(alpha=0.3)

    # 类别数量趋势
    axes[2].plot(time_steps, category_counts, '^-', color='green')
    axes[2].set_title('类别数量随时间变化')
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('类别数')
    axes[2].set_xticks(time_steps)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('temporal_trends.png', dpi=300)
    plt.show()

    # 打印关键统计信息
    print("\n=== 时序图数据统计信息 ===")
    print(f"时间步范围: 0 ~ {time_steps[-1]}")
    print(f"节点数变化: {node_counts[0]} → {node_counts[-1]} (增长 {node_counts[-1]/node_counts[0]:.2f} 倍)")
    print(f"边数变化: {edge_counts[0]} → {edge_counts[-1]} (增长 {edge_counts[-1]/edge_counts[0]:.2f} 倍)")
    print(f"类别数变化: {category_counts[0]} → {category_counts[-1]}")

def visualize_sequential_graphs(all_graphs: List[Dict], step_interval: int = 2):
    """可视化按时间序列选取的图结构"""
    # 按间隔选取时间步进行可视化
    selected_steps = all_graphs[::step_interval]
    n_plots = len(selected_steps)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for i, graph in enumerate(selected_steps):
        t = graph["time_step"]
        print(f"可视化时间步 {t}: 节点数 {graph['num_nodes']}, 边数 {len(graph['edges'])}")

        # 采样节点以避免图过于密集（最多显示50个节点）
        sample_size = min(50, graph["num_nodes"])
        sample_indices = np.random.choice(graph["num_nodes"], sample_size, replace=False)
        sample_node_ids = graph["node_ids"][sample_indices]

        # 筛选采样节点之间的边
        sample_edges = []
        node_id_to_idx = {id: idx for idx, id in enumerate(graph["node_ids"])}
        for (u, v) in graph["edges"]:
            if u in sample_node_ids and v in sample_node_ids:
                sample_edges.append((u, v))

        # 创建图并绘制
        G = nx.Graph()
        G.add_nodes_from(sample_node_ids)
        G.add_edges_from(sample_edges)

        # 按类别着色
        node_cats = graph["node_categories"][sample_indices]
        nx.draw(
            G, ax=axes[i],
            node_size=60, node_color=node_cats, cmap=plt.cm.tab10,
            edge_color='#999999', edge_width=0.5, with_labels=False
        )

        axes[i].set_title(f'时间步 {t}\n(类别数: {graph["num_categories"]})')

    plt.tight_layout()
    plt.savefig('sequential_graphs_visualization.png', dpi=300)
    plt.show()

def edges_to_csr(edges):
    """
    将边列表转换为csr_matrix格式的邻接矩阵

    参数:
        edges: 边列表，每个元素为包含两个节点ID的元组

    返回:
        adj: csr_matrix格式的邻接矩阵
        node_mapping: 节点ID到矩阵索引的映射字典
    """
    # 提取所有唯一节点
    nodes = []
    for u, v in edges:
        nodes.append(u)
        nodes.append(v)
    unique_nodes = np.unique(nodes)
    num_nodes = len(unique_nodes)
    print("num_nodes:", num_nodes)
    # 创建节点ID到矩阵索引的映射（将节点ID转换为0,1,2...的连续索引）
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

    # 准备构建邻接矩阵的数据
    row_indices = []
    col_indices = []
    data = []

    # 遍历边列表，填充数据
    for u, v in edges:
        # 将节点ID转换为矩阵索引
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]

        # 无向图：添加双向边
        row_indices.append(u_idx)
        col_indices.append(v_idx)
        data.append(1)  # 边的权重，这里默认为1

        row_indices.append(v_idx)
        col_indices.append(u_idx)
        data.append(1)

    # 构建csr矩阵
    adj = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(num_nodes, num_nodes)
    )

    return adj, node_mapping


def train_density_gmm(X, y, k=None, reg_lambda=1e-6):
    """
    Train a 2-component 1D GMM on density changes for each class.

    Args:
        X (torch.Tensor): Feature matrix of shape (n_samples, n_features).
        y (torch.Tensor): Labels of shape (n_samples,).
        k (int, optional): Number of classes. If None, inferred from y.
        reg_lambda (float): Regularization for covariance matrix.

    Returns:
        dict: GMM parameters {'weights': ..., 'means': ..., 'vars': ...}
    """
    n_samples, n_features = X.shape

    # 推断k（如果未提供）
    if k is None:
        k = torch.unique(y).numel()

    # 函数：计算密度 (-logdet(cov + reg * I))
    def get_density(points):
        if points.shape[0] < 2:
            return torch.tensor(0.0, device=points.device)
        cov = torch.cov(points.T)
        cov_reg = cov + reg_lambda * torch.eye(n_features, device=points.device)
        log_det = torch.logdet(cov_reg)
        return -log_det

    # 第一步：分组数据并计算每个类的原始密度
    class_to_points = {}
    class_to_mask = {}
    class_densities = {}
    unique_classes = torch.unique(y)
    for label in unique_classes:
        label = label.item()
        mask = (y == label)
        class_to_mask[label] = mask
        points = X[mask]
        class_to_points[label] = points
        class_densities[label] = get_density(points)

    # 第二步：收集正负样本的密度变化
    positive_changes = []
    negative_changes = []
    for i in range(n_samples):
        c = y[i].item()
        point = X[i:i+1]

        # 移除i后的类c点集
        mask_without_i = class_to_mask[c] & (torch.arange(n_samples, device=X.device) != i)
        points_without_i = X[mask_without_i]
        density_without = get_density(points_without_i)

        for j in unique_classes:
            j = j.item()
            if j == c:
                # 添加回c
                new_points = torch.cat((points_without_i, point), dim=0)
                density_new = get_density(new_points)
                change = density_new - density_without
                positive_changes.append(change)
            else:
                # 添加到j
                points_j = class_to_points[j]
                density_original_j = class_densities[j]
                new_points = torch.cat((points_j, point), dim=0)
                density_new = get_density(new_points)
                change = density_new - density_original_j
                negative_changes.append(change)

    # 第三步：收集所有变化，训练GMM
    all_changes = torch.cat((torch.tensor(positive_changes, device=X.device),
                             torch.tensor(negative_changes, device=X.device)))
    all_changes = all_changes.view(-1, 1)  # 确保为 (n_samples, 1)

    # GMM类（1D，2成分）
    class GMM1D:
        def __init__(self, n_components=2, max_iter=100, tol=1e-4):
            self.n_components = n_components
            self.max_iter = max_iter
            self.tol = tol

        def fit(self, data):
            n_samples = data.shape[0]
            # 确保data是二维张量 (n_samples, 1)
            if data.dim() == 1:
                data = data.view(-1, 1)

            # 初始化
            self.weights = torch.ones(self.n_components, device=data.device) / self.n_components
            self.means = torch.tensor([data[:, 0].min(), data[:, 0].max()], device=data.device)
            self.vars = torch.ones(self.n_components, device=data.device) * data[:, 0].var()

            prev_log_lik = -float('inf')
            for _ in range(self.max_iter):
                # E步
                log_prob = torch.zeros(n_samples, self.n_components, device=data.device)
                for k in range(self.n_components):
                    dist = torch.distributions.Normal(self.means[k], torch.sqrt(self.vars[k]))
                    log_prob[:, k] = dist.log_prob(data[:, 0]) + torch.log(self.weights[k])
                log_prob_norm = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
                resp = torch.exp(log_prob_norm)

                # M步
                nk = resp.sum(dim=0)
                self.weights = nk / n_samples
                self.means = (resp.T @ data[:, 0]) / nk  # 标量均值
                for k in range(self.n_components):
                    diff = data[:, 0] - self.means[k]  # 正确广播
                    self.vars[k] = (resp[:, k] * (diff ** 2)).sum() / nk[k]

                # 检查收敛
                log_lik = torch.sum(torch.logsumexp(log_prob, dim=1))
                if abs(log_lik - prev_log_lik) < self.tol:
                    break
                prev_log_lik = log_lik

            return {'weights': self.weights, 'means': self.means, 'vars': self.vars}

    # 训练GMM并返回
    gmm = GMM1D(n_components=2)
    gmm_params = gmm.fit(all_changes)
    return gmm_params


def classify_new_vector(new_feat, class_to_points, class_densities, gmm_params, reg_lambda=1e-6, threshold=0.5):
    """
    将新特征向量加入每个类，计算密度变化，使用变化方差和GMM判断是否为已知类。

    参数:
        new_feat (torch.Tensor): 新特征向量，形状为 [500] 或 [1, 500]。
        class_to_points (dict): 每个类的点集 {label: tensor [n_j, 500]}。
        class_densities (dict): 每个类的原始密度 {label: float}。
        gmm_params (dict): GMM参数 {'weights': tensor(2,), 'means': tensor(2,), 'vars': tensor(2,)}。
        reg_lambda (float): 正则化参数，默认1e-6。
        threshold (float): 正概率阈值，默认0.5。

    返回:
        result (str): "Known class: <label>" 或 "Unknown class"。
        max_pos_prob (float): 最大正概率。
        var_changes (float): 所有变化的方差。
        changes (list): 每个类的密度变化列表。
    """
    # GMM类（1D）
    class GMM1D:
        def __init__(self, weights, means, vars):
            self.weights = weights
            self.means = means
            self.vars = vars
            self.n_components = len(weights)

        def log_prob(self, data):
            log_prob = torch.zeros(data.shape[0], self.n_components, device=data.device)
            for k in range(self.n_components):
                dist = torch.distributions.Normal(self.means[k], torch.sqrt(self.vars[k]))
                log_prob[:, k] = dist.log_prob(data.squeeze()) + torch.log(self.weights[k])
            return log_prob

        def posterior(self, data):
            log_prob = self.log_prob(data)
            log_prob_norm = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
            return torch.exp(log_prob_norm)

    # 实例化GMM
    gmm = GMM1D(gmm_params['weights'], gmm_params['means'], gmm_params['vars'])

    # 密度计算函数
    def get_density(points):
        if points.shape[0] < 2:
            return torch.tensor(0.0, device=points.device)
        cov = torch.cov(points.T)
        n_features = points.shape[1]
        cov_reg = cov + reg_lambda * torch.eye(n_features, device=points.device)
        log_det = torch.logdet(cov_reg)
        return -log_det

    # 确保new_feat是[1, 500]
    new_feat = new_feat.view(1, -1).to(next(iter(class_to_points.values())).device)

    # 计算每个类的密度变化
    changes = []
    for j in class_to_points.keys():
        points_j = class_to_points[j]
        density_original = class_densities[j]
        new_points = torch.cat((points_j, new_feat), dim=0)
        density_new = get_density(new_points)
        change = density_new - density_original
        changes.append(change.item())

    # 变化张量
    changes_tensor = torch.tensor(changes, device=new_feat.device).view(-1, 1)

    # 计算后验概率
    posteriors = gmm.posterior(changes_tensor)

    # 识别正成分（假设均值较大的是正样本成分）
    pos_idx = torch.argmax(gmm.means).item()
    pos_probs = posteriors[:, pos_idx]

    # 变化方差
    var_changes = np.var(changes)

    # 判断
    max_pos_prob = pos_probs.max().item()
    if max_pos_prob > threshold:
        predicted_class = list(class_to_points.keys())[pos_probs.argmax().item()]
        result = f"Known class: {predicted_class}"
    else:
        result = "Unknown class"

    return result, max_pos_prob, var_changes, changes


def classify_new_vector(new_feat, features, labels, gmm_params, reg_lambda=1e-6, threshold=0.5):
    """
    将新特征向量加入每个类，计算密度变化，使用变化方差和GMM判断是否为已知类。

    参数:
        new_feat (torch.Tensor): 新特征向量，形状为 [n_features] 或 [1, n_features]。
        features (torch.Tensor): 老数据特征矩阵，形状为 [n_samples, n_features]。
        labels (torch.Tensor): 老数据标签，形状为 [n_samples,]。
        gmm_params (dict): GMM参数 {'weights': tensor(2,), 'means': tensor(2,), 'vars': tensor(2,)}。
        reg_lambda (float): 正则化参数，默认1e-6。
        threshold (float): 正概率阈值，默认0.5。

    返回:
        result (str): "Known class: <label>" 或 "Unknown class"。
        max_pos_prob (float): 最大正概率。
        var_changes (float): 所有变化的方差。
        changes (list): 每个类的密度变化列表。
    """
    # GMM类（1D）
    class GMM1D:
        def __init__(self, weights, means, vars):
            self.weights = weights
            self.means = means
            self.vars = vars
            self.n_components = len(weights)

        def log_prob(self, data):
            log_prob = torch.zeros(data.shape[0], self.n_components, device=data.device)
            for k in range(self.n_components):
                dist = torch.distributions.Normal(self.means[k], torch.sqrt(self.vars[k]))
                log_prob[:, k] = dist.log_prob(data.squeeze()) + torch.log(self.weights[k])
            return log_prob

        def posterior(self, data):
            log_prob = self.log_prob(data)
            log_prob_norm = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
            return torch.exp(log_prob_norm)

    # 实例化GMM
    gmm = GMM1D(gmm_params['weights'], gmm_params['means'], gmm_params['vars'])

    # 密度计算函数
    def get_density(points):
        if points.shape[0] < 2:
            return torch.tensor(0.0, device=points.device)
        cov = torch.cov(points.T)
        n_features = points.shape[1]
        cov_reg = cov + reg_lambda * torch.eye(n_features, device=points.device)
        log_det = torch.logdet(cov_reg)
        return -log_det

    # 确保new_feat是[1, n_features]
    new_feat = new_feat.view(1, -1).to(features.device)
    if new_feat.shape[1] != features.shape[1]:
        raise ValueError(f"new_feat dimension {new_feat.shape[1]} does not match features dimension {features.shape[1]}")

    # 分组数据并计算每个类的原始密度
    class_to_points = {}
    class_densities = {}
    labels = torch.tensor(labels, dtype=torch.int)
    unique_classes = torch.unique(labels)
    for label in unique_classes:
        label = label.item()
        mask = (labels == label)
        points = features[mask]
        class_to_points[label] = points
        class_densities[label] = get_density(points)

    # 计算每个类的密度变化
    changes = []
    for j in class_to_points.keys():
        points_j = class_to_points[j]
        density_original = class_densities[j]
        new_points = torch.cat((points_j, new_feat), dim=0)
        density_new = get_density(new_points)
        change = density_new - density_original
        changes.append(change.item())

    # 变化张量
    changes_tensor = torch.tensor(changes, device=features.device).view(-1, 1)

    # 计算后验概率
    posteriors = gmm.posterior(changes_tensor)

    # 识别正成分（假设均值较大的是正样本成分）
    pos_idx = torch.argmax(gmm.means).item()
    pos_probs = posteriors[:, pos_idx]

    # 变化方差
    var_changes = np.var(changes) if len(changes) > 1 else 0.0

    # 判断
    max_pos_prob = pos_probs.max().item()
    predicted_class = -1
    if max_pos_prob > threshold:
        predicted_class = list(class_to_points.keys())[pos_probs.argmax().item()]
        # 这是已知类
        result = 0
    else:
        # 这是未知类
        result = 1

    return result, max_pos_prob, var_changes, changes, predicted_class

def generate_and_select(inx, indice, num_nearest=100, num_generate=100, sigma=0.01, eta=0.01, epochs=100, batch_size=32, topk = 50):
    """
    以inx[indice]为中心，取最近的num_nearest个样本为训练集，使用MLP模拟diffusion模型生成num_generate个样本。
    噪声程度sigma=0.01。在去噪过程中使用核密度函数(KDE)引导去噪方向。
    最后计算生成样本和最近样本到中心的距离，并排序。取出前50个中存在的最近样本的下标。

    参数:
        inx (torch.Tensor): 特征矩阵，形状为 [N, D]。
        indice (int): 中心样本的索引。
        num_nearest (int): 最近样本数，默认100。
        num_generate (int): 生成样本数，默认100。
        sigma (float): 噪声程度，默认0.01。
        eta (float): KDE引导步长，默认0.01。
        epochs (int): MLP训练轮数，默认100。
        batch_size (int): 批次大小，默认32。

    返回:
        list: 前50个排序中存在的最近样本的下标列表（已排序）。
    """

    N, D = inx.shape
    center = inx[indice]
    distances = torch.norm(inx - center, dim=1)
    _, nearest_indices = torch.topk(distances, num_nearest, largest=False)
    nearest_indices = nearest_indices.sort().values  # 排序以便后续映射
    training_data = inx[nearest_indices]

    # MLP for noise prediction
    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, d)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(D).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练MLP
    for epoch in range(epochs):
        for b in range((len(training_data) // batch_size) + 1):
            start = b * batch_size
            end = min(start + batch_size, len(training_data))
            if start >= end:
                break
            x0 = training_data[start:end]
            noise = torch.randn_like(x0)
            x_t = x0 + sigma * noise
            predicted_noise = model(x_t)
            loss = nn.MSELoss()(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # KDE score function
    def kde_score(x, data, h):
        diff = x[:, None, :] - data[None, :, :]
        log_k = -0.5 * (diff ** 2).sum(dim=-1) / (h ** 2)
        w = torch.softmax(log_k, dim=-1)
        score = (w @ data - x) / (h ** 2)
        return score

    h = (D / 2.0) ** 0.5  # bandwidth approximation

    # 生成样本
    generated = []
    for _ in range(num_generate):
        x_t = sigma * torch.randn(1, D, device=inx.device)
        predicted_noise = model(x_t)
        x_gen = x_t - sigma * predicted_noise
        score = kde_score(x_gen, training_data, h)
        x_gen += eta * score  # 引导
        generated.append(x_gen)

    generated = torch.cat(generated, dim=0)

    # 所有样本（最近 + 生成）
    all_samples = torch.cat((training_data, generated), dim=0)
    all_distances = torch.norm(all_samples - center, dim=1)
    sorted_indices = torch.argsort(all_distances)
    top50 = sorted_indices[:topk]
    nearest_in_top50_local = top50[top50 < num_nearest]
    original_indices = nearest_indices[nearest_in_top50_local]
    return sorted(original_indices.tolist())

def classify_test_set_mlp(inx, f_labels, classes=7, hidden_dim=256, epochs=100, batch_size=32, lr=1e-3, seed=64):
    """
    使用MLP对测试集（f_labels == -1）进行分类，并更新f_labels。

    参数:
        inx (torch.Tensor): 特征矩阵，形状 [N, D]。
        f_labels (torch.Tensor): 标签矩阵，形状 [N,]，-1表示测试集，其他值为类别。
        hidden_dim (int): MLP隐藏层维度，默认256。
        epochs (int): 训练轮数，默认100。
        batch_size (int): 批次大小，默认32。
        lr (float): 学习率，默认1e-3。
        seed (int): 随机种子，默认64。

    返回:
        torch.Tensor: 更新后的f_labels，测试集位置填入预测类别。
    """
    torch.manual_seed(seed)  # 确保可复现
    device = inx.device
    f_labels = f_labels.to(device, dtype=torch.long)

    # 分离训练集和测试集
    train_mask = f_labels != -1
    test_mask = f_labels == -1
    train_indices = torch.where(train_mask)[0]
    test_indices = torch.where(test_mask)[0]

    if len(test_indices) == 0:
        print("没有测试集样本（f_labels == -1）")
        return f_labels
    if len(train_indices) == 0:
        raise ValueError("没有训练集样本（f_labels != -1）")

    train_features = inx[train_mask]
    train_labels = f_labels[train_mask]
    test_features = inx[test_mask]

    # 推断类别数
    num_classes = classes

    # 定义MLP模型
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    # 初始化模型
    model = MLP(input_dim=inx.shape[1], hidden_dim=hidden_dim, output_dim=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练MLP
    model.train()
    for epoch in range(epochs):
        for b in range((len(train_features) // batch_size) + 1):
            start = b * batch_size
            end = min(start + batch_size, len(train_features))
            if start >= end:
                break
            x_batch = train_features[start:end]
            y_batch = train_labels[start:end]
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 预测测试集
    model.eval()
    with torch.no_grad():
        logits = model(test_features)
        predicted_labels = torch.argmax(logits, dim=1)

    # 更新f_labels
    f_labels[test_indices] = predicted_labels

    print(f"分类完成：{len(test_indices)}个测试样本已分配类别")
    print(f"更新后f_labels中的类别分布: {torch.bincount(f_labels[f_labels != -1])}")

    return f_labels


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    # acc=1
    # f1=1
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # if show_details:
    #     print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #           ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1



class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv_mid = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x_em = x
        x = torch.relu(x)
        # x = self.conv_mid(x, edge_index)
        # x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return x, x_em


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def createData(X, y, A):
    features = X

    adj = sp.csr_matrix(A)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    num_nodes = features.shape[0]
    y = torch.tensor(y, dtype=torch.long)
    unique_classes = torch.unique(y)
    num_classes = unique_classes.numel()

    # 1. 从 csr_matrix 中提取边的信息
    rows, cols = adj.nonzero()  # 获取非零元素的行和列索引
    # 2. 将边的信息转换为 edge_index 格式
    adj = torch.tensor([rows, cols], dtype=torch.long)
    sm_fea_s = sp.csr_matrix(features).toarray()
    sm_fea_s = torch.tensor(sm_fea_s, dtype=torch.float)
    # 4. 掩码 (随机生成示例数据)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.8 * num_nodes)] = 1  # 前80%节点作为训练集
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[int(0.8 * num_nodes):int(0.9 * num_nodes)] = 1  # 接下来10%节点作为验证集
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[int(0.9 * num_nodes):] = 1  # 最后10%节点作为测试集

    dataset = Data(x=sm_fea_s, y=y, edge_index=adj, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_classes=num_classes)
    data = dataset.clone().detach()
    return data

def train(data, classes, labels, mask):
    for epoch in range(400):
        gnn_model = GCNModel(data.x.shape[1], 256, classes)
        gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
        gnn_optimizer.zero_grad()
        # gcn_output, gmm_output = integrated_model(data.x, data.edge_index)
        gcn_output, gcn_output_em = gnn_model(data.x, data.edge_index)
        # print("gmm_output:", gmm_output.shape)
        # 计算GCN的损失
        loss_gcn = criterion_gmm(gcn_output[mask], labels[mask])
        total_loss = loss_gcn
        total_loss.backward()
        gnn_optimizer.step()

        if epoch%1 == 0:

            # 评估模型
            gnn_model.eval()
            with torch.no_grad():
                # 得到生成的特征向量
                logits, gcn_output_em = gnn_model(data.x, data.edge_index)
            # 计算准确率
            _, predicted = logits.max(dim=1)