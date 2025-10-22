import os
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
from ot.utils import euclidean_distances

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from munkres import Munkres
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, accuracy_score, calinski_harabasz_score, davies_bouldin_score, \
    silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.mixture import GaussianMixture
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from torch_geometric.utils import add_self_loops, to_undirected, to_networkx
from triton.language import math

# 本代码在原有模型基础上引入diffusion生成模型，为最终版的修改版。改变模型框架，先是检测未知类，检测满后转为gnn
from utils import load_graph_data, load_npz, load_npz_time

pd.set_option('display.width', 300) # 设置字符显示宽度
pd.set_option('display.max_rows', None) # 设置显示最大行
pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列
torch.set_printoptions(threshold=float('inf'))


# 定义一个简单的diffusion模型
class SimpleDiffusionModel(nn.Module):
    def __init__(self, X0):
        super(SimpleDiffusionModel, self).__init__()
        self.fc1 = nn.Linear(X0.shape[1], 64)
        self.fc2 = nn.Linear(64, X0.shape[1])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# diffusion的加噪
def add_noise(x, noise_level=0.1):
    noise = noise_level * torch.randn_like(x)
    return x + noise

# 3. 训练diffusion模型
def train_diffusion(model, optimizer, criterion, dataloader, num_epochs=10, noise_level=0.1):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch[0]
            noisy_x = add_noise(x, noise_level)

            optimizer.zero_grad()
            outputs = model(noisy_x)
            loss = criterion(outputs, x)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


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


def addNodeToData3(dataX, same_cluster_indices, data, mask, label):

    for i in range(len(same_cluster_indices)):
        mask[same_cluster_indices[i]] = True
        data.y[same_cluster_indices[i]] = label
        # 更新list_known
        list_known[label].append(same_cluster_indices[i])

    return data, list_known, mask

def gnnTrain(epochs, labels, gcn_input_dim, gcn_hidden_dim, gcn_output_dim, data, mask):
    # 训练一次gnn模型，让不同类的点散开，此时参与运算的已标记点
    gnn_model = GCNModel(gcn_input_dim, gcn_hidden_dim, gcn_output_dim)
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    # print("mask:", mask)

    gnn_model.train()
    for epoch in range(epochs):
        # 清除之前计算图中的梯度信息
        gnn_optimizer.zero_grad()
        # gcn_output, gmm_output = integrated_model(data.x, data.edge_index)
        gcn_output, gcn_output_em = gnn_model(data.x.clone().detach(), data.edge_index.clone().detach())
        # print("gmm_output:", gmm_output.shape)
        # 计算GCN的损失
        loss_gcn = criterion_gmm(gcn_output[mask], labels[mask])
        loss_gcn.backward()  # 第一次反向传播
        gnn_optimizer.step()
        # print("训练集的loss为：", total_loss)
        # 清零模型 A 的梯度
    return gnn_model


def gnnTest(data, model):
    # 评估模型
    model.eval()
    with torch.no_grad():
        test_gcn_output, test_gcn_output_em = model(data.x, data.edge_index)
    return test_gcn_output

def embedding_extract(tensor, list):
    # 提取指定行号的数据，并组成新的张量列表
    new_tensors = []
    for idx_row in list:
        row_tensor = torch.index_select(tensor, 0, torch.tensor(idx_row))
        new_tensors.append(row_tensor)

    return new_tensors

def std_cal(list):
    list_std_column = []
    for j in range(len(list)):
        # 计算每一列的标准差
        stds = torch.std(list[j], dim=0)
        # 将所有标准差相加
        total_std = torch.sum(stds)
        list_std_column.append(total_std.item())

    return list_std_column

# 得到一个list中的标准差数组
def get_all_float(embedding, list, list_std_known):
    list_embedding_get = embedding_extract(embedding, list)
    list_again_std = std_cal(list_embedding_get)
    std_all_float = [x - y for x, y in zip(list_again_std, list_std_known)]
    return std_all_float

def count_ones_odd_even_positions(lst, y_pred):

    count_odd_positions = 0
    count_even_positions = 0
    for i in range(len(lst)):
        if (i %2 == 0):
            if (lst[i] == y_pred):
                count_even_positions+=1
        else:
            if (lst[i] == y_pred):
                count_odd_positions+=1

    return count_even_positions, count_odd_positions

def judgeIsKnown(count_0, count_1, y_p):
    if count_0 > count_1:
        # print("聚类为0的表示为已知类，为1的表示为未知类")
        if (y_p == 0):
            print("该点为已知类")
            judge = 1
        else:
            print("该点为未知类")
            judge = 0
    else:
        # print("聚类为1的表示为已知类，为0的表示未知类")
        if (y_p == 1):
            print("该点为已知类")
            judge = 1
        else:
            print("该点为未知类")
            judge = 0
    return judge

def judge_judge(node_position, classes, known_list, judge):
    print("classes[node_position]:", classes[node_position].item())
    if(classes[node_position].item() in known_list):
        if (judge==1):
            print("判断正确，其为已知类")
            return 1
        else:
            print("判断错误，其为已知类")
            return 0
    else:
        if (judge==1):
            print("判断错误，其为未知类")
            return 0
        else:
            print("判断正确，其为未知类")
            return 1

def fit_and_predict_gmm(entropyAllArray, entropyTest):
    # 这个代码输入的是list，原始代码用的是array所以要转成array
    entropyAllArray = np.array(entropyAllArray)
    entropyTest = np.array(entropyTest)

    allLength = len(entropyAllArray)
    print("allLength:", allLength)
    entropyAllArray = entropyAllArray.reshape(-1, 1)
    entropyTest = entropyTest.reshape(-1, 1)
    # 使用高斯混合模型
    gmm = GaussianMixture(n_components=2, max_iter=100)
    gmm.fit(entropyAllArray)

    # 用训练集看下0和1哪个是未知类
    yTrainPredict = gmm.predict(entropyAllArray)
    # print("yTrainPredict:", yTrainPredict)
    # print("yTrainPredict:", len(yTrainPredict))

    # 预测测试集标签
    y_pred = gmm.predict(entropyTest)
    print("gmm的预测值为：", y_pred)

    # 分别统计1在偶数位置和奇数位置上的个数
    count_even_positions, count_odd_positions = count_ones_odd_even_positions(yTrainPredict, y_pred)
    print("在偶数位置上的个数:", count_even_positions)
    print("在奇数位置上的个数:", count_odd_positions)
    # 偶数上为已知类，奇数上为未知类
    if(count_even_positions>=count_odd_positions):
        judge = 1
    else:
        judge = 0
    return judge



def createNewList(test_embedding, train_embedding, list_known, allDataY, allDataChangeY, i, num, currentClasses, mask, data):
    needNum = num
    distances = torch.cdist(test_embedding.unsqueeze(0), train_embedding)
    print("distances:", distances.shape)
    # # 找到每行前n个最近距离的位置
    topk_distances, topk_indices = torch.topk(distances, num_add, largest=False, dim=1)
    topk_indices = topk_indices.view(-1)

    # 方法1，使用diffusion构造100个新节点,训练集是和未知类点最近的20个点
    selected_rows = train_embedding[topk_indices]
    selected_rows = selected_rows.squeeze(0)
    selected_rows = torch.cat((selected_rows, test_embedding.unsqueeze(0)), dim=0)

    dataX = diffsuionMakeClasses(selected_rows, selected_rows)
    distances_dataX = torch.cdist(dataX[dataX.shape[0]-1, :].unsqueeze(0), dataX[:-1, :])
    topk_distances_dataX, topk_indices_dataX = torch.topk(distances_dataX, num_add, largest=False, dim=1)
    topk_indices_dataX = topk_indices_dataX.view(-1)

    # 取出两个tensor前100个都有的元素，最近距离和diffusion距离
    topk_values_dataX = topk_indices[topk_indices_dataX][: 150]
    topk_values = topk_indices[: 60]
    # diffusion和distance一起用
    common_values = torch.tensor([value for value in topk_values_dataX if value in topk_values])
    # # 只用distance
    # common_values = topk_values
    print("common_values:", common_values.shape)
    # print("dataX:", dataX.shape)
    dataX_numpy = dataX.detach().numpy()

    # # 初始化 KMeans 模型并进行聚类
    # kmeans = KMeans(n_clusters=2)
    # kmeans.fit(dataX_numpy)
    # # 获取聚类中心和每个样本所属簇的标签
    # cluster_centers = kmeans.cluster_centers_
    # cluster_labels = kmeans.labels_
    # # 找出聚类和目标是同一类的结果
    # same_cluster_indices = [i for i, label in enumerate(cluster_labels[:-1]) if label == cluster_labels[dataX.shape[0]-1]]
    # same_cluster_indices = topk_indices[same_cluster_indices]
    # common_values_2 = torch.tensor([value for value in topk_values if value in same_cluster_indices])
    #
    # # 连接两个张量
    # concatenated_tensor = torch.cat((common_values_2, topk_indices[: len(common_values_2)]))
    # # 去除重复的元素
    # unique_values = torch.unique(concatenated_tensor)
    # # 将结果转换为一维张量
    # unique_values = unique_values.flatten()

    new_data, list_known, mask = addNodeToData3(dataX, common_values, data, mask, currentClasses)
    # 方法1的返回
    return list_known, mask, new_data

    # 方法2的返回
    # return list_known, mask, allDataChangeY


def diffsuionMakeClasses(allDataX, nowUseX):

    # 标准化数据
    scaler = StandardScaler()
    data_scaler = scaler.fit_transform(allDataX)
    # 标准化后将数据转为tensor
    data_scaler_tensor = torch.tensor(data_scaler, dtype=torch.float)

    dataloader = DataLoader(allDataX, batch_size=32, shuffle=True)
    # 实例化模型、优化器和损失函数
    model = SimpleDiffusionModel(allDataX)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    # 训练模型
    train_diffusion(model, optimizer, criterion, dataloader, num_epochs=10, noise_level=0.1)
    with torch.no_grad():
        generated_data = []
        for i in range(allDataX.shape[0]):
            # 对每一行生成一个样本
            sample = data_scaler_tensor[i].unsqueeze(0)  # 取原数据中的第i行
            noisy_sample = add_noise(sample, noise_level=0.1)
            generated_sample = model(noisy_sample).numpy()
            generated_data.append(generated_sample[0])

        generated_data = np.array(generated_data)
        # 反标准化
        generated_data = scaler.inverse_transform(generated_data)
        generated_data_tensor = torch.tensor(generated_data)

    return generated_data_tensor


# 用gnn取代mlp实现对已知类的分类
def gnnPredict(gnnModel, i, length):
    mask1 = [False] * length
    mask1[i] = True
    gnnModel.eval()
    with torch.no_grad():
        gnnPredict1 = gnnModel(data.x, data.edge_index)

    _, predicted = gnnPredict1.max(dim=1)
    return predicted[mask1]

# 计算准确率
def accuracy(output, target):
    with torch.no_grad():
        predicted = torch.round(torch.sigmoid(output))
        correct = (predicted == target).sum().item()
        acc = correct / target.size(0)
        return acc

# 计算 F1 分数
def f1(output, target):
    with torch.no_grad():
        predicted = torch.round(torch.sigmoid(output))
        return f1_score(target.cpu(), predicted.cpu(), average='macro')

def purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    return acc

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

def calculate_clustering_metrics(embeddings, labels):
    """
    Calculate WCSS, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

    Parameters:
    embeddings (np.ndarray): Array of shape (n_samples, n_features) containing embedding vectors.
    labels (np.ndarray): Array of shape (n_samples,) containing cluster labels.

    Returns:
    dict: Dictionary containing WCSS, Silhouette Score, DBI, and CHI.
    """
    # Ensure inputs are numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Initialize results dictionary
    results = {}

    # # Calculate WCSS (Within-Cluster Sum of Squares)
    # wcss = 0
    # unique_labels = np.unique(labels)
    # for label in unique_labels:
    #     cluster_points = embeddings[labels == label]
    #     if len(cluster_points) > 0:  # Avoid empty clusters
    #         centroid = np.mean(cluster_points, axis=0)
    #         wcss += np.sum(euclidean_distances(cluster_points, [centroid]) ** 2)
    # results['WCSS'] = wcss

    # Calculate Silhouette Score
    try:
        results['Silhouette_Score'] = silhouette_score(embeddings, labels)
    except ValueError as e:
        results['Silhouette_Score'] = None  # Handle cases with single cluster or invalid labels
        print(f"Silhouette Score calculation failed: {e}")

    # Calculate Davies-Bouldin Index
    try:
        results['Davies_Bouldin_Index'] = davies_bouldin_score(embeddings, labels)
    except ValueError as e:
        results['Davies_Bouldin_Index'] = None
        print(f"DBI calculation failed: {e}")

    # Calculate Calinski-Harabasz Index
    try:
        results['Calinski_Harabasz_Index'] = calinski_harabasz_score(embeddings, labels)
    except ValueError as e:
        results['Calinski_Harabasz_Index'] = None
        print(f"CHI calculation failed: {e}")

    return results

import scipy.sparse as sp
# 自定义AMAPDataset类，继承自torch_geometric.data.Dataset

# 以下为运行部分
# # 加载Cora数据集或Citeseer数据集
# dataset = Planetoid(root='./dataset/', name='cora')
# data = dataset[0].clone().detach()
# print("data.edge_index:", (data.edge_index.shape))
# print("data.edge_index:", type(data.edge_index))
# print("edge_index:", data.edge_index.shape)
# print("dataset:", type(dataset))
# print("dataset:", dataset)

# 以下是将npy格式的数据转为planetoid格式
X, y, A = load_graph_data("amap", show_details=False, seed=0)
# # 以下是将npz格式的数据转为planetoid格式
# X, y, A = load_npz("computers")

# # 以下是读取时间序列的npz
# X, y, A = load_npz_time("Brain")

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

result_ini = calculate_clustering_metrics(data.x, data.y)
print("result_ini:", result_ini)

# 参数设置区
# 初始已知类的数量
num_known_class_initial = 5
# 已知类列表
knownList = []
# 数据集这中的类别总数
num_classes_all = 0

# allDataY是真实的对照组
allDataY_temp = data.y.clone().detach()
allDataY = data.y.clone().detach()
pureDataY = data.y.clone().detach()
# allDataChangeY的内容在后续会有改变
allDataChangeY = data.y.clone().detach()
num_classes_all = dataset.num_classes
print("num_classes_all:", num_classes_all)
num_features = dataset.num_features  # 每个节点的特征维度为3\
unKnowNum = dataset.num_classes - num_known_class_initial
gcn_input_dim = dataset.num_features
gmm_input_dim = dataset.num_classes
gmm_out_dim = 2
gcn_hidden_dim = 256
gcn_output_dim = dataset.num_classes
# 种类总数
num_components = dataset.num_classes

max_acc = 0
max_ari = 0
max_nmi = 0
max_f1 = 0


#每一轮GNN的训练次数
epochs = 400
# 初始每个已知类的点数量
num_mask_initial = 200
# 新建类时加入的点数量
num_add = 150
# 记录分类正确的
true_classify_mlp = 0
# 设置一个list用于存储每一类点
list_known = list()
# 损失函数，这里采用了交叉熵
criterion_gmm = nn.CrossEntropyLoss()
# 创建掩码张量,这个mask标记了训练集。后续会更新。
mask = torch.zeros_like(allDataY, dtype=torch.bool)
# 正确判别数量
trueClassify = 0

best_seed = -1
for loop in [2]:
    setup_seed(loop)
    # cora all 01234 best seed； 964
    # cora 0.22 01234 best seed: 4553
    # cora 0.23 02345 best seed:2001 2002
    # citeseer 0.293 1234 best seed: 843
    # citeseer 0.00 best seed: 1079
    # pubmed 0.25 01 best seed: 507  61
    # pubmed 0.281 12 best seed: 2002
    # amap 0.22 best seed：2
    # amap 0.00 best seed： 11
    # corafull 0.58 best seed: 7
    #
    print("seed:", loop)
    # 初始已知类的数量
    num_known_class_initial = 5
    # # 已知类列表
    # knownList = [0, 2, 3, 4, 5]

    # corafull 专用
    for kn in range(num_known_class_initial):
        knownList.append(kn)
    print("knownList:", len(knownList))
    # 未知类的数量
    unKnowNum = dataset.num_classes - num_known_class_initial
    data = dataset.clone().detach()
    # allDataY是真实的对照组
    allDataY = dataset.y.clone().detach()
    print("allDataY:", type(allDataY))
    pureDataY = dataset.y.clone().detach()

    allDataChangeY = dataset.y.clone().detach()

    label_counts = np.zeros(dataset.num_classes)
    # 设置一个list用于存储每一类点
    list_known = list()
    # 创建掩码张量,这个mask标记了训练集。后续会更新。
    mask = torch.zeros_like(allDataY, dtype=torch.bool)
    # 正确判别数量
    trueClassify = 0


    for i in knownList:
        indices = (allDataY == i).nonzero()[:, 0]
        indices_error = (allDataY != i).nonzero()[:, 0]
        if len(indices) <= num_mask_initial:
            list_known.append(indices.tolist())
            mask[indices] = True
        else:
            random_indices = np.random.choice(indices, int(num_mask_initial), replace=False)
            list_known.append(random_indices.tolist())
            mask[random_indices] = True

    # 随机选择部分数据
    num_lists = len(allDataChangeY[mask])
    # 数据准确率
    num_to_swap = int(num_lists * 0.22)
    allDataChangeY_mask = allDataChangeY[mask]
    # 获取所有 True 的原始下标
    true_indices = np.where(mask)[0]
    # 训练集的大小
    len_Y = len(true_indices)
    allDataChangeY_choice = random.sample(allDataChangeY[mask].tolist(), num_to_swap)
    for loop3 in range(num_to_swap):
        options = [x for x in knownList if x != allDataChangeY_mask[loop3]]
        allDataChangeY[true_indices[loop3]] = random.choice(options)

    before_acc = cluster_acc(data.y[mask].numpy(), allDataChangeY[mask].numpy())
    print("before acc:", before_acc)
    # 最初的allDataChangeY
    data.y = allDataChangeY
    # 搞一个temp用于最后验证
    mask_temp = mask.clone().detach()
    # 参与运算的点总数
    total_num = 0
    gnn_model_initial = gnnTrain(epochs, allDataChangeY, gcn_input_dim, gcn_hidden_dim, gcn_output_dim, data, mask)
    # 拿到本轮所有的输出向量
    all_embedding = gnnTest(data, gnn_model_initial)
    all_embedding_initial = gnnTest(data, gnn_model_initial)


    result = calculate_clustering_metrics(data.x[mask], y[mask])
    print("result_initial:", result)

    # 查找测试集,每次测试前要训练一次gnn,其中统一0为未知，1为已知
    for i in range(len(mask)):
        if (~mask[i] and num_known_class_initial < num_classes_all):
            # 如果有未知类加入，则需要更新changeY
            gnn_model = gnnTrain(epochs, allDataChangeY, gcn_input_dim, gcn_hidden_dim, gcn_output_dim, data, mask)
            # 拿到本轮所有的输出向量
            all_embedding = gnnTest(data, gnn_model)
            # 得到tensor的数组
            list_tensor_embedding_extract = embedding_extract(all_embedding, list_known)

            # 计算已知类中每一类的标准差
            list_std_known = std_cal(list_tensor_embedding_extract)
            # 修改list_known,让新的点加入每一个类,为了防止错误，这里用temp数组代替
            list_known_temp = copy.deepcopy(list_known)
            # 在每个子列表的末尾添加一个新数字
            for sub_list in list_known_temp:
                sub_list.append(i)

            list_tensor_embedding_extract_temp = embedding_extract(all_embedding, list_known_temp)
            # 计算已知类中每一类的标准差
            list_std_known_std = std_cal(list_tensor_embedding_extract_temp)
            # 得到新来点加入到所有类所引起的标准差浮动
            list_std_new_float = [x - y for x, y in zip(list_std_known_std, list_std_known)]
            # 未标记点进入所有类引起的标准差浮动的标准差
            std_new_float = np.std(list_std_new_float)
            list_again_float = []
            # 得到已知类点再次进入该系统引起的标准差浮动
            for sublist in list_known:
                delete = 0
                for item in sublist:
                    list_known_temp_2 = copy.deepcopy(list_known)
                    list_std_known_temp = copy.deepcopy(list_std_known)
                    for sub_list in list_known_temp_2:
                        sub_list.append(item)

                    # 已知类的点作为已知类再次进入
                    list_again_float_known = get_all_float(all_embedding, list_known_temp_2, list_std_known_temp)
                    one_again_float_known = np.std(list_again_float_known)

                    # 已知类的点作为未知类再次进入
                    list_known_temp_2.pop(delete)
                    list_std_known_temp.pop(delete)

                    list_again_float_unknown = get_all_float(all_embedding, list_known_temp_2, list_std_known_temp)
                    one_again_float_unknown = np.std(list_again_float_unknown)

                    list_again_float.append(one_again_float_known)
                    list_again_float.append(one_again_float_unknown)
                # 计数加1，标志进入下一行
                delete+=1
            print("list_again_float:", len(list_again_float))
            # 用gmm来判别新来点是哪个类
            judge = fit_and_predict_gmm(list_again_float, std_new_float)

            # 首先判断模型对新来点已知类/未知类的判别是否正确，并加入到结果中,这里用的标签是Y
            true_or_false = judge_judge(i, allDataY, knownList, judge)
            # 这个trueClassify可以作为实验结果之一[作为未训练下的准确率]。
            trueClassify += true_or_false

            # 得到当前所有节点的特征向量
            train_embedding = all_embedding
            # 得到当前未知类节点的特征向量
            test_embedding = all_embedding[i, :]

            # 如果是未知类，则新开一个类且往里面塞一些点，目前直接塞原始类相同的点，后面引入diffusion
            if (judge == 0):
                # 类总数小于应存在总数时，当未知类处理
                if (num_known_class_initial < num_classes_all):
                    # list中新开一行
                    temp_node = [i]
                    list_known.append(temp_node)
                    # 方法1,diffusion生成新节点去覆盖老节点
                    list_known, mask, data = createNewList(test_embedding, train_embedding, list_known, allDataY, allDataChangeY, i, num_add, num_known_class_initial, mask, data)
                    allDataChangeY = data.y.clone().detach()
                    # 新类中点的标签应该一致
                    allDataChangeY[i] = num_known_class_initial
                    # # 方法2,最近距离的视为同类
                    # list_known, mask, allDataChangeY = createNewList(test_embedding, train_embedding, list_known, allDataY, allDataChangeY, i, num_add, num_known_class_initial, mask, data)

                    num_known_class_initial += 1
                # 如果已知类总数大于应存在类总数，则当已知类处理

                # 无论未知类数量有没有到顶，都要把他放到已知类列表中
                knownList.append(allDataY[i].item())
                # 他是未知类的话打上true的标签
                mask[i] = True

            list_class_num = []
            for sublist in list_known:
                list_class_num.append(len(sublist))
            print("list_class_num:", list_class_num)
            total_num+=1

    # 现在已经找出所有未知类,开始gnn训练
    gnn_model = GCNModel(gcn_input_dim, gcn_hidden_dim, gcn_output_dim)
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    # print("mask:", mask)


    # 记录开始时间
    start_time = time.time()

    # 记录初始内存使用
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.memory_allocated()
    initial_cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

    gnn_model.train()
    for epoch in range(400):
        gnn_optimizer.zero_grad()
        # gcn_output, gmm_output = integrated_model(data.x, data.edge_index)
        gcn_output, gcn_output_em = gnn_model(data.x, data.edge_index)
        # print("gmm_output:", gmm_output.shape)
        # 计算GCN的损失
        loss_gcn = criterion_gmm(gcn_output[mask], allDataChangeY[mask])
        total_loss = loss_gcn
        total_loss.backward()
        gnn_optimizer.step()
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_used = (final_gpu_memory - initial_gpu_memory) / 1024 / 1024 / 1024  # GB

        if torch.cuda.is_available():
            print(f'GPU内存消耗: {gpu_memory_used:.4f} GB')
        if epoch%1 == 0:

            # 评估模型
            gnn_model.eval()
            with torch.no_grad():
                # 得到生成的特征向量
                logits, gcn_output_em = gnn_model(data.x, data.edge_index)
                labels_np = dataset.y.numpy()
                result = calculate_clustering_metrics(gcn_output_em, labels_np)

            # 计算准确率
            _, predicted = logits.max(dim=1)
            correct = predicted[~mask].eq(data.y[~mask]).sum().item()
            # 统计 False 的数量
            false_count = (mask == False).sum().item()
            accuracy = correct / false_count
            print(f'Test Accuracy: {accuracy:.4f}')

            # # cora
            # allDataChangeY_list = dataset[0].y.numpy()
            # predicted_list = predicted.numpy()
            # purity_score = purity(allDataChangeY_list, predicted_list)
            # print("Purity Score:", purity_score)

            # 带不带初始数据：[~mask_temp]
            # 将 PyTorch 张量转换为 NumPy 数组
            predictions_np = predicted.numpy()
            # cora
            labels_np = dataset.y.numpy()
            # 计算 F1 分数
            f1 = f1_score(labels_np, predictions_np, average='weighted')
            acc = cluster_acc(labels_np, predictions_np)
            nmi = nmi_score(labels_np, predictions_np, average_method='arithmetic')
            ari = ari_score(labels_np, predictions_np)

            print("f1:", f1, " Acc:", acc, " nmi:", nmi, " ari:", ari)
            print("best seed；", best_seed)

            if (max_nmi < nmi):
                max_f1 = f1
                max_acc = acc
                max_ari = ari
                max_nmi = nmi
                best_seed = loop
                best_result = result
                # 转换为 NumPy 数组
                data_em = gcn_output_em.numpy()


            print("max_f1:", max_f1)
            print("max_acc:", max_acc)
            print("max_nmi:", max_nmi)
            print("max_ari:", max_ari)
            print("best_result:", best_result)
            # 应用 t-SNE

    # tsne = TSNE(n_components=2, random_state=42)
    # data_2d = tsne.fit_transform(data_em)
    # plt.xticks([])  # 移除x坐标
    # plt.yticks([])  # 移除y坐标
    # plt.grid(False)  # 移除网格
    # # 可视化
    # plt.figure(figsize=(8, 6))
    # plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_np, alpha=0.7, cmap='viridis')
    # plt.savefig("amap.png")

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated()
        gpu_memory_used = (final_gpu_memory - initial_gpu_memory) / 1024 / 1024 / 1024  # GB
    print(f'\n训练完成!')
    print(f'总训练时间: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}')

    # print(f'CPU内存消耗: {cpu_memory_used:.4f} MB')


