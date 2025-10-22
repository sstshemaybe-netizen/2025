import os
import argparse
import pickle

import numpy as np
import pandas as pd
import psutil
import torch
import matplotlib
from sklearn.metrics import silhouette_score
from torch_geometric.data import Data

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.stats import gaussian_kde
from sklearn.cluster import SpectralClustering, Birch
from sklearn.decomposition import PCA
from torch.nn import Parameter
from sklearn.manifold import TSNE
from utils import *
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import time

import warnings

pd.set_option('display.width', 300) # 设置字符显示宽度
pd.set_option('display.max_rows', None) # 设置显示最大行
pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列
torch.set_printoptions(threshold=float('inf'))


# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# 本代码用于将flow模型转变为无k聚类模型
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--k', type=float, default=1, help='pass of Filter.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--num_flows', type=int, default=10)
parser.add_argument('--epsilon', type=float, default=0.5)
parser.add_argument('--out', type=int, default=500)
parser.add_argument('--cluster_num', type=int, default=7)
parser.add_argument('--centers', type=int, default=10)
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--sigma', type=int, default=1)
parser.add_argument('--eta', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--norm', type=str, default="sym")
parser.add_argument('--mu', type=int, default=0)
parser.add_argument('--pca', type=int, default=500)
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--init_class', type=float, default=5)
parser.add_argument('--batch_size', type=int, default=500)

setup_seed(1)
args = parser.parse_args()

# 参数区
for args.dataset in ["corafull"]:
    print("Using {} dataset".format(args.dataset))
    if args.dataset == 'cora':
        args.hidden = 256
        args.sigma = 1
        args.eta = 1
        args.gnnlayers = 3
        args.lr = 1e-4
        args.num_flows = 10
        args.ratio = 0.3
        args.init_class = 5
        args.add_ratio = 0.1
        args.classes = 7
        args.batch_size = 171
    if args.dataset == 'corafull':
        args.hidden = 256
        args.sigma = 1
        args.eta = 1
        args.gnnlayers = 3
        args.lr = 1e-4
        args.num_flows = 10
        args.ratio = 0.3
        args.init_class = 65
        args.add_ratio = 0.1
        args.classes = 70
        args.batch_size = 12578

    # load data
    if args.dataset in ['computers', 'photo', 'phy']:
        X, y, A = load_npz(args.dataset)
    else:
        X, y, A = load_graph_data(args.dataset, show_details=False, seed=0)

    # if args.dataset in ['Brain', 'DBLP3', 'Brain_re']:
    #     X, y, A = load_npz_time(args.dataset)

    features = X
    print("A:", type(A))
    print("原始X:", features.shape)
    # 记录原始坐标的list
    list_pos = list(range(features.shape[0]))
    true_labels = y
    true_labels_numpy = torch.from_numpy(true_labels)
    print("true_labels_numpy:", true_labels_numpy.shape)
    print("true_labels:", type(true_labels))
    adj = sp.csr_matrix(A)

    features = sp.csr_matrix(features).toarray()
    features = torch.tensor(features, dtype=torch.float)
    # all是受污染的标签
    all_labels = np.copy(true_labels)
    f_features, f_adj, f_labels, current_indices, remaining_indices, selected_labels = initialize_dataset(features, adj, true_labels, args.init_class, args.ratio)

    # label打乱作为其他的输出
    f_labels, f_mask = corrupt_labels_within_range(f_labels, error_ratio=0.27, random_seed=64)
    # all_labels[current_indices] = f_labels
    f_labels = torch.tensor(f_labels, dtype=torch.int).to(args.device)
    pca = PCA(n_components=args.pca, random_state=64)
    f_features = pca.fit_transform(f_features)
    f_features = torch.FloatTensor(f_features)


    f_adj = f_adj - sp.dia_matrix((f_adj.diagonal()[np.newaxis, :], [0]), shape=f_adj.shape)
    f_adj.eliminate_zeros()
    print("f_adj:", f_adj.shape)
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(f_adj, args.gnnlayers, norm=args.norm, renorm=True, k=args.k)
    sm_fea_s = sp.csr_matrix(f_features).toarray()

    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    adj_1st = (f_adj + sp.eye(f_adj.shape[0])).toarray()

    sm_fea_s = torch.tensor(sm_fea_s, dtype=torch.float)
    inx = sm_fea_s.to(args.device)
    print("inx.shape:", inx.shape)

    # 存储每一步的结果（可选）
    incremental_results = []

    # 初始数据加入结果列表
    # incremental_results.append((init_features, init_A, init_labels))

    # 获取剩余样本中属于已选类别的数量（用于循环条件）
    remaining_mask = np.isin(true_labels[remaining_indices], selected_labels)
    valid_remaining_count = np.sum(remaining_mask)

    while len(remaining_indices) > 0:
        # 用老数据训练GMM
        old_inx_indix = inx.shape[0]
        print("inx:", inx.shape)
        print("f_labels:", f_labels.shape)
        f_labels = torch.tensor(f_labels, dtype=torch.int).to(args.device)
        GMM = train_density_gmm(inx, f_labels, k=None, reg_lambda=1e-6)

        # 加入新数据
        f_features, f_adj, f_labels, current_indices, remaining_indices, added, add_indices = add_incremental_data(features, adj, f_labels, all_labels, current_indices, remaining_indices, selected_labels, args.batch_size)
        # 如果没有新加入的节点 则退出循环
        if added == 0:
            break


        f_adj = f_adj - sp.dia_matrix((f_adj.diagonal()[np.newaxis, :], [0]), shape=f_adj.shape)
        f_adj.eliminate_zeros()
        print("f_adj:", f_adj.shape)
        print('Laplacian Smoothing...')
        adj_norm_s = preprocess_graph(f_adj, args.gnnlayers, norm=args.norm, renorm=True, k=args.k)
        sm_fea_s = sp.csr_matrix(f_features).toarray()

        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        adj_1st = (f_adj + sp.eye(f_adj.shape[0])).toarray()

        sm_fea_s = torch.tensor(sm_fea_s, dtype=torch.float)
        inx = sm_fea_s.to(args.device)
        print("new inx.shape:", inx.shape)
        inx_come = torch.zeros(inx.shape[0], dtype=torch.long)
        for indice in range(old_inx_indix, inx.shape[0]):
            if inx_come[indice] == 0:
                # 打标签使其跳过
                inx_come[indice] = 1
                if args.init_class < args.classes:
                    # print("inx[:old_inx_indix, :]:", inx[:old_inx_indix, :].shape)
                    result, max_pos_prob, var_changes, changes, predicted_class = classify_new_vector(inx[indice], inx[:old_inx_indix, :], f_labels[:old_inx_indix, ], GMM)
                else:
                    continue
                if result == 0:
                    f_labels[indice] = predicted_class
                else:
                    # 发现未知类
                    if args.init_class < args.classes:
                        available_labels = np.setdiff1d(np.arange(args.classes), selected_labels)
                        print("available_labels:", available_labels)
                        selected_label = np.random.choice(available_labels)
                        # 新加一类
                        selected_labels.append(selected_label)
                        f_labels[indice] = selected_label
                        # 现在利用diffusion进行样本扩充
                        o_indices = generate_and_select(inx, indice, num_nearest=150, batch_size=32, topk=100)
                        f_labels[o_indices] = selected_label
                        inx_come[o_indices] = 1
                        args.init_class += 1
                    else:
                        continue
            else:
                continue

    features = X

    pca = PCA(n_components=args.pca, random_state=64)
    features = pca.fit_transform(features)
    features = torch.FloatTensor(features)

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
    true_labels = torch.tensor(true_labels, dtype=torch.long)
    # 现在已经找出所有未知类,开始gnn训练
    gnn_model = GCNModel(data.x.shape[1], 256, args.classes)
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    # print("mask:", mask)
    labels = torch.zeros_like(f_labels)
    labels[current_indices] = f_labels

    labels = torch.tensor(labels, dtype=torch.long)
    mask = labels != -1
    mask = mask.to(dtype=torch.bool, device=labels.device)
    gnn_model.train()
    # 损失函数，这里采用了交叉熵
    criterion_gmm = nn.CrossEntropyLoss()
    max_nmi = 0

    # 记录开始时间
    start_time = time.time()

    # 记录初始内存使用
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.memory_allocated()
    initial_cpu_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

    for epoch in range(400):
        epoch_start = time.time()
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
            acc, nmi, ari, f1 = eva(true_labels, predicted.cpu().numpy())
            if (max_nmi < nmi):
                max_f1 = f1
                max_acc = acc
                max_ari = ari
                max_nmi = nmi
                # 转换为 NumPy 数组
                data_em = gcn_output_em.numpy()
    # 计算总训练时间
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # 计算内存消耗
    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated()
        gpu_memory_used = (final_gpu_memory - initial_gpu_memory) / 1024 / 1024 / 1024  # GB
    print(f'\n训练完成!')
    print(f'总训练时间: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}')
    if torch.cuda.is_available():
        print(f'GPU内存消耗: {gpu_memory_used:.4f} MB')
    # print(f'CPU内存消耗: {cpu_memory_used:.4f} MB')



print("max_f1:", max_f1)
print("max_acc:", max_acc)
print("max_nmi:", max_nmi)
print("max_ari:", max_ari)

