import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as utils
from sklearn.metrics import adjusted_rand_score as ari_score, normalized_mutual_info_score as nmi_score

from utils import load_graph_data

X, y, A = load_graph_data("cora", show_details=False, seed=0)
adj = torch.tensor(A, dtype=torch.float)  # Changed to float for numerical stability

# Simulate dynamic with same adj for T=3
T = 12
adj_mats = torch.stack([adj.clone() for _ in range(T)], dim=0)  # (T, N, N)

class DyGMF:
    def __init__(self, num_clusters_rho=7, embed_dim_r=128, num_subsets_s=10, dynamic_ratio_mu=0.16,
                 reg_beta=20.0, temp_lambda=0.2, eps=1e-6, max_iters=20, lr=0.001, device=None):
        self.rho = num_clusters_rho
        self.r = embed_dim_r
        self.s = num_subsets_s
        self.mu = dynamic_ratio_mu
        self.beta = reg_beta
        self.lambda_ = temp_lambda
        self.eps = eps  # Increased eps for stability
        self.max_iters = max_iters
        self.lr = lr
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def compute_pmi(self, adj):
        adj = adj.to(self.device)
        degrees = adj.sum(dim=1) + self.eps
        vol = degrees.sum() / 2.0
        pmi = torch.log(vol * adj / (degrees.unsqueeze(1) * degrees.unsqueeze(0) + self.eps) + self.eps)
        pmi = torch.clamp(pmi, min=0.0)
        return pmi

    def select_landmarks(self, M_t, M_prev=None):
        torch.manual_seed(42)
        N = M_t.shape[0]
        num_landmarks = int(0.5 * N)
        if M_prev is None:
            centers = M_t[torch.randperm(N, device=self.device)[:self.rho]]
            for _ in range(5):
                dists = torch.cdist(M_t, centers, p=2)
                assignments = torch.argmin(dists, dim=1)
                for l in range(self.rho):
                    mask = assignments == l
                    if mask.sum() > 0:
                        centers[l] = M_t[mask].mean(dim=0)
        else:
            combined = M_t + self.lambda_ * M_prev
            factor = 1 + self.lambda_
            centers = M_t[torch.randperm(N, device=self.device)[:self.rho]]
            for _ in range(5):
                dists = torch.cdist(combined / factor, centers, p=2)
                assignments = torch.argmin(dists, dim=1)
                for l in range(self.rho):
                    mask = assignments == l
                    if mask.sum() > 0:
                        centers[l] = combined[mask].sum(dim=0) / (factor * mask.sum())
        dists = torch.zeros(N, device=self.device)
        assignments = torch.zeros(N, dtype=torch.long, device=self.device)
        for l in range(self.rho):
            dist_l = torch.norm(M_t - centers[l], dim=1, p=2)**2
            if M_prev is not None:
                dist_l += self.lambda_ * torch.norm(M_prev - centers[l], dim=1, p=2)**2
            if l == 0:
                dists = dist_l
                assignments.fill_(l)
            mask = dist_l < dists
            dists[mask] = dist_l[mask]
            assignments[mask] = l
        U_t = []
        num_per = max(1, num_landmarks // self.rho)
        for l in range(self.rho):
            dist_l = torch.norm(M_t - centers[l], dim=1, p=2)**2
            if M_prev is not None:
                dist_l += self.lambda_ * torch.norm(M_prev - centers[l], dim=1, p=2)**2
            _, indices = torch.topk(dist_l, num_per, largest=False)
            U_t.append(indices)
        U_t = [t.flatten() for t in U_t]
        U_t = torch.cat(U_t, dim=0)
        U_t = torch.unique(U_t)
        U_t = torch.sort(U_t).values
        U_t = torch.tensor(U_t, dtype=torch.long, device=self.device)
        return U_t, centers, assignments

    def bi_clustering_reg(self, C_i):
        m, r = C_i.shape
        # Ensure C_i is non-negative
        C_i = torch.clamp(C_i, min=0.0)
        zero_r = torch.zeros(r, r, device=self.device)
        zero_m = torch.zeros(m, m, device=self.device)
        S = torch.cat((torch.cat((zero_r, C_i.T), dim=1), torch.cat((C_i, zero_m), dim=1)), dim=0)
        degrees = S.sum(dim=1) + self.eps
        # Add additional regularization to degrees
        degrees = torch.clamp(degrees, min=self.eps)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))
        L = torch.eye(S.shape[0], device=self.device) - D_inv_sqrt @ S @ D_inv_sqrt
        # Ensure L is symmetric and semi-definite
        L = (L + L.T) / 2.0  # Symmetrize
        try:
            vals = torch.linalg.eigvalsh(L)
            reg = vals[:self.rho].sum()
        except RuntimeError as e:
            print(f"Warning: Eigenvalue computation failed: {e}. Returning default reg value.")
            reg = torch.tensor(0.0, device=self.device)  # Fallback to avoid crashing
        return reg

    def nmf(self, M, rank, iters=50):
        N = M.shape[0]
        W = torch.rand(N, rank, device=self.device) + self.eps
        H = torch.rand(rank, N, device=self.device) + self.eps
        for _ in range(iters):
            W = W * (M @ H.T) / (W @ H @ H.T + self.eps)
            W = F.normalize(W, p=1, dim=1)
            H = H * (W.T @ M) / (W.T @ W @ H + self.eps)
        return W, H

    def fit(self, adj_mats):
        T = adj_mats.shape[0]
        clusters = []
        C_prev = None
        Phi_prev = None
        Psi_prev = None
        theta_prev = None
        for t in range(T):
            adj_t = adj_mats[t].to(self.device)
            adj_prev = adj_mats[t-1].to(self.device) if t > 0 else None
            adj_next = adj_mats[t+1].to(self.device) if t < T-1 else None
            M_t = self.compute_pmi(adj_t)
            M_prev = self.compute_pmi(adj_prev) if adj_prev is not None else None
            U_t, centers, assignments = self.select_landmarks(M_t, M_prev)
            theta_t = centers[assignments]
            M00 = M_t[U_t[:, None], U_t]
            if t == 0:
                Phi_t, Psi_t = self.nmf(M00, self.r)
                X_t = torch.arange(X.shape[0], device=self.device)
                Y_t = torch.tensor([], dtype=torch.long, device=self.device)
            else:
                if adj_next is None:
                    avg_w = (adj_prev + adj_t) / 2.0
                    avg_theta = (theta_prev + theta_t) / 2.0
                else:
                    avg_w = (adj_prev + adj_t + adj_next) / 3.0
                    avg_theta = (theta_prev + theta_t + theta_t) / 3.0
                delta = torch.norm(adj_t - avg_w, dim=1)**2 + torch.norm((adj_t - theta_t) - (avg_w - avg_theta), dim=1)**2
                num_dynamic = int(self.mu * X.shape[0])
                _, idx = torch.topk(delta, num_dynamic, largest=True)
                X_t = idx.sort()[0]
                full_nodes = torch.arange(X.shape[0], device=self.device)
                remaining = full_nodes[~torch.isin(full_nodes, U_t)]
                Y_t = full_nodes[~torch.isin(full_nodes, X_t)]
                idx_u_x = torch.isin(U_t, X_t)
                idx_u_y = ~idx_u_x
                U_x = U_t[idx_u_x]
                U_y = U_t[idx_u_y]
                idx_map = torch.zeros(X.shape[0], dtype=torch.long, device=self.device)
                idx_map[U_t] = torch.arange(len(U_t), device=self.device)
                idx_u_x = idx_map[U_x]
                idx_u_y = idx_map[U_y]
                Phi_y = Phi_prev[idx_u_y]
                Psi_y = Psi_prev[:, idx_u_y]
                Phi_x = torch.rand(len(U_x), self.r, device=self.device) + self.eps
                Psi_x = torch.rand(self.r, len(U_x), device=self.device) + self.eps
                Phi_x.requires_grad = True
                Psi_x.requires_grad = True
                opt = torch.optim.Adam([Phi_x, Psi_x], lr=self.lr)
                for _ in range(self.max_iters * 2):
                    opt.zero_grad()
                    loss = 0.0
                    M_xx = M00[idx_u_x[:, None], idx_u_x]
                    recon_xx = Phi_x @ Psi_x
                    loss += (recon_xx - M_xx).pow(2).sum()
                    if len(U_y) > 0:
                        M_xy = M00[idx_u_x[:, None], idx_u_y]
                        recon_xy = Phi_x @ Psi_y
                        loss += (recon_xy - M_xy).pow(2).sum()
                        M_yx = M00[idx_u_y[:, None], idx_u_x]
                        recon_yx = Phi_y @ Psi_x
                        loss += (recon_yx - M_yx).pow(2).sum()
                        M_yy = M00[idx_u_y[:, None], idx_u_y]
                        recon_yy = Phi_y @ Psi_y
                        loss += (recon_yy - M_yy).pow(2).sum()
                    loss.backward()
                    opt.step()
                    Phi_x.data.clamp_(min=self.eps)
                    Psi_x.data.clamp_(min=self.eps)
                Phi_t = torch.zeros(len(U_t), self.r, device=self.device)
                Psi_t = torch.zeros(self.r, len(U_t), device=self.device)
                Phi_t[idx_u_x] = Phi_x.detach()
                Phi_t[idx_u_y] = Phi_y
                Psi_t[:, idx_u_x] = Psi_x.detach()
                Psi_t[:, idx_u_y] = Psi_y
            Phi_t = F.normalize(Phi_t, p=1, dim=1)
            full_nodes = torch.arange(X.shape[0], device=self.device)
            remaining = full_nodes[~torch.isin(full_nodes, U_t)]
            Y_t = full_nodes[~torch.isin(full_nodes, X_t)]
            perm = torch.randperm(len(remaining), device=self.device)
            remaining = remaining[perm]
            subsets = torch.chunk(remaining, self.s)
            gamma_list = [g for g in subsets if len(g) > 0]
            C_t = torch.zeros(X.shape[0], self.r, device=self.device)
            for gamma_i in gamma_list:
                idx_i_x = torch.isin(gamma_i, X_t)
                gamma_i_x = gamma_i[idx_i_x]
                gamma_i_y = gamma_i[~idx_i_x]
                len_x = len(gamma_i_x)
                len_y = len(gamma_i_y)
                if t == 0 or len_y == 0:
                    P_i = torch.rand(len(gamma_i), len(U_t), device=self.device) + self.eps
                    Q_i = torch.rand(len(U_t), len(gamma_i), device=self.device) + self.eps
                else:
                    C_i_y = C_prev[gamma_i_y]
                    P_i_y = C_i_y @ torch.pinverse(Phi_t)
                    Q_i_y = P_i_y.T
                    P_i_x = torch.rand(len_x, len(U_t), device=self.device) + self.eps
                    Q_i_x = torch.rand(len(U_t), len_x, device=self.device) + self.eps
                    P_i = torch.zeros(len(gamma_i), len(U_t), device=self.device)
                    P_i[idx_i_x] = P_i_x
                    P_i[~idx_i_x] = P_i_y
                    Q_i = torch.zeros(len(U_t), len(gamma_i), device=self.device)
                    Q_i[:, idx_i_x] = Q_i_x
                    Q_i[:, ~idx_i_x] = Q_i_y
                if len_x > 0:
                    P_i_x = P_i[idx_i_x].clone().detach().requires_grad_(True)
                    Q_i_x = Q_i[:, idx_i_x].clone().detach().requires_grad_(True)
                    opt = torch.optim.Adam([P_i_x, Q_i_x], lr=self.lr)
                    for _ in range(self.max_iters):
                        opt.zero_grad()
                        loss = 0.0
                        M_ii_xx = M_t[gamma_i_x[:, None], gamma_i_x]
                        recon_xx = P_i_x @ M00 @ Q_i_x
                        loss += (recon_xx - M_ii_xx).pow(2).sum()
                        if len_y > 0:
                            M_ii_xy = M_t[gamma_i_x[:, None], gamma_i_y]
                            recon_xy = P_i_x @ M00 @ Q_i_y
                            loss += (recon_xy - M_ii_xy).pow(2).sum()
                            M_ii_yx = M_t[gamma_i_y[:, None], gamma_i_x]
                            recon_yx = P_i_y @ M00 @ Q_i_x
                            loss += (recon_yx - M_ii_yx).pow(2).sum()
                            M_ii_yy = M_t[gamma_i_y[:, None], gamma_i_y]
                            recon_yy = P_i_y @ M00 @ Q_i_y
                            loss += (recon_yy - M_ii_yy).pow(2).sum()
                        M_i0_x = M_t[gamma_i_x[:, None], U_t]
                        recon_i0_x = P_i_x @ M00
                        loss += (recon_i0_x - M_i0_x).pow(2).sum()
                        if len_y > 0:
                            M_i0_y = M_t[gamma_i_y[:, None], U_t]
                            recon_i0_y = P_i_y @ M00
                            loss += (recon_i0_y - M_i0_y).pow(2).sum()
                        M_0i_x = M_t[U_t[:, None], gamma_i_x]
                        recon_0i_x = M00 @ Q_i_x
                        loss += (recon_0i_x - M_0i_x).pow(2).sum()
                        if len_y > 0:
                            M_0i_y = M_t[U_t[:, None], gamma_i_y]
                            recon_0i_y = M00 @ Q_i_y
                            loss += (recon_0i_y - M_0i_y).pow(2).sum()
                        C_i = P_i @ Phi_t
                        br_reg = self.bi_clustering_reg(C_i)
                        loss += self.beta * br_reg
                        loss.backward()
                        opt.step()
                        P_i_x.data.clamp_(min=self.eps)
                        Q_i_x.data.clamp_(min=self.eps)
                        C_i = P_i @ Phi_t
                        C_i = F.normalize(C_i, p=1, dim=1)
                        P_i = C_i @ torch.pinverse(Phi_t)
                        Q_i = P_i.T
                        P_i[idx_i_x] = P_i_x.detach()
                        Q_i[:, idx_i_x] = Q_i_x.detach()
                else:
                    C_i = P_i @ Phi_t
                    C_i = F.normalize(C_i, p=1, dim=1)
                C_t[gamma_i] = C_i
            kmeans = KMeans(n_clusters=self.rho, random_state=42)
            cluster_t = kmeans.fit_predict(C_t.cpu().numpy())
            cluster_t = torch.tensor(cluster_t, dtype=torch.long, device=self.device)
            clusters.append(cluster_t)
            C_prev = C_t
            Phi_prev = Phi_t
            Psi_prev = Psi_t
            theta_prev = theta_t
        return clusters

def eva(y_true, y_pred, show_details=True):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}")
    return nmi, ari

# Usage
model = DyGMF(num_clusters_rho=7)
clusters = model.fit(adj_mats)
print("Clusters for timestamp 0:", clusters[0])
print("Unique cluster labels:", torch.unique(clusters[0]))
nmi, ari = eva(y, clusters[-1].cpu().numpy())
print("NMI:", nmi)
print("ARI:", ari)