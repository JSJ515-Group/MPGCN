import math
import torch_geometric
import world
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


# 个简单的矩阵分解模型，用于隐式反馈数据
class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.interactionGraph = self.dataset.getInteractionGraph()
        print(f"{world.model_name} is already to go")

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        G = self.interactionGraph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(G, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)  # 将每一层的嵌入矩阵堆叠在一起
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss


class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, num_layers=3, dropout=0.4):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.fc_layers = nn.ModuleList()

        # 第一层输入到隐藏层
        self.fc_layers.append(nn.Linear(in_features, hidden_size))

        # 中间层
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

        # 输出层
        self.fc_out = nn.Linear(hidden_size, out_features)

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.residual_layers = nn.ModuleList([nn.Linear(in_features, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        residual = x  # 初始化残差

        for i in range(self.num_layers):
            # 计算当前层的输出
            x = self.fc_layers[i](x)

            # 激活函数（ReLU）
            x = F.leaky_relu(x, negative_slope=0.2)

            # Dropout
            x = self.dropout(x)
            if i != self.num_layers - 1:
                x = x + self.residual_layers[i](residual)

        # 输出层
        x = self.fc_out(x)

        return x


class GumbelSoftmaxClustering(nn.Module):
    def __init__(self, in_features, out_features, num_clusters, temperature, hidden_size=128):
        super(GumbelSoftmaxClustering, self).__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.mlp = MLP(in_features, hidden_size, out_features)
        self.cluster_weights = nn.Parameter(torch.randn(in_features, num_clusters))  # 初始化聚类权重

    def forward(self, node_embeddings):
        node_embeddings = self.mlp(node_embeddings)
        logits = torch.matmul(node_embeddings, self.cluster_weights)  # 计算每个节点对各个聚类的分配概率
        gumbel_softmax_out = F.gumbel_softmax(logits, tau=self.temperature, hard=False)  # 软采样
        return gumbel_softmax_out  # 返回每个节点的软分配概率


class GatedGraphConv(nn.Module):
    def __init__(self, in_features, out_features, num_users, num_items, num_clusters=3, temperature=1.0):
        super(GatedGraphConv, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.gumbel_softmax = GumbelSoftmaxClustering(in_features, out_features, num_clusters, temperature)

    def forward(self, node_embeddings, adj):

        cluster_probs = self.gumbel_softmax(node_embeddings)  # [batch_size, num_clusters]

        # 初始化子图输出列表
        subgraph_outs = []

        # 对每个聚类进行处理
        for c in range(self.num_clusters):
            # 获取邻接矩阵中每条边的连接节点
            node_i, node_j = adj._indices()  # 获取邻接矩阵的边索引

            # 获取每个用户节点和物品节点在当前聚类中的软分配概率
            prob_u = cluster_probs[node_i, c]  # 用户节点 i 在聚类 c 中的概率
            prob_i = cluster_probs[node_j, c]  # 物品节点 j 在聚类 c 中的概率

            # 对每条边的加权值进行处理：使用边的连接节点的概率
            cluster_adj_values = adj._values() * prob_u * prob_i

            # 创建独立的邻接矩阵，用于当前聚类的图
            cluster_adj = torch.sparse_coo_tensor(adj._indices(), cluster_adj_values, adj.size(), device=adj.device)

            # 对该聚类特定的图进行图卷积
            h = torch.sparse.mm(cluster_adj, node_embeddings)

            subgraph_outs.append(h)

        # 聚合每个聚类的输出
        final_output = torch.stack(subgraph_outs, dim=1)  # 将每个子图的输出堆叠
        final_output = torch.sum(final_output, dim=1)  # 对所有聚类的输出求和

        return final_output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.att = nn.Parameter(torch.FloatTensor(out_features * 2, 1))
        nn.init.xavier_uniform_(self.att.data, gain=1.414)

    def forward(self, h, adj):

        h_i = h[adj._indices()[0]]
        h_j = h[adj._indices()[1]]

        # 注意力得分计算
        attention_input = torch.cat([h_i, h_j], dim=1)
        attention = F.leaky_relu(torch.matmul(attention_input, self.att).squeeze(1), negative_slope=0.2)

        # softmax 归一化
        attention = F.softmax(attention, dim=0)
        attention = F.dropout(attention, 0.3, training=self.training)  # Dropout

        # 更新邻接矩阵的值
        adj_values = adj._values() * attention
        new_adj = torch.sparse_coo_tensor(adj._indices(), adj_values, adj.size(), device=adj.device)

        # 计算图卷积结果
        h_prime = torch.sparse.mm(new_adj, h)

        return h_prime


class MPGCN(LightGCN):
    def _init_weight(self):
        super(MPGCN, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb(self.latent_dim)
        self.gat_layer = GraphAttentionLayer(self.latent_dim, self.latent_dim)
        self.conv = GatedGraphConv(self.latent_dim, self.latent_dim, self.num_users, self.num_items, num_clusters=2)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        A = self.interactionGraph
        S = self.socialGraph
        embs = [all_emb]
        temp_all_emb = torch.sparse.mm(A, all_emb)
        all_emb = all_emb + temp_all_emb

        for layer in range(self.n_layers):

            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])

            users_emb_social = self.gat_layer(users_emb, S)

            all_emb_interaction = self.conv(all_emb, A)

            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])

            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items


# 用于融合两个不同嵌入（x 和 y）的神经网络模块。它在 MPGCN 模型中用于将社交网络嵌入和交互网络嵌入进行融合。
class Graph_Comb(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, num_layers=4, dropout=0.5):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)

        # 动态加权参数
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # MLP层设置
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(embed_dim, hidden_dim))  # 第一层调整为 hidden_dim
        self.norm_layers = nn.ModuleList()
        self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers - 1):
            self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

        self.residual_transform = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))  # (batch_size, embed_dim)
        h2 = torch.tanh(self.att_y(y))  # (batch_size, embed_dim)

        # 动态加权融合
        combined = self.alpha * h1 + (1 - self.alpha) * h2  # 融合后的维度为 (batch_size, embed_dim)

        residual = self.residual_transform(combined)  # (batch_size, hidden_dim)

        for layer, norm_layer in zip(self.mlp_layers, self.norm_layers):
            new_combined = F.relu(layer(combined))
            combined = norm_layer(new_combined + residual)
            combined = self.dropout(combined)
            residual = combined

        output = self.output_layer(combined)  # 输出转换为 embed_dim
        output = output / output.norm(2, dim=1, keepdim=True)  # 正则化
        return output
