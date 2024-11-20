import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import add_self_loops


class HeteroGCNEncoder(torch.nn.Module):
    """
    HeteroGCN编码器类
    基于HeteroGCN对图中节点进行编码
    """
    def __init__(self, metadata, hidden_dim, num_layer,embedding_dim):
        super(HeteroGCNEncoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        # 初始化每层的异构卷积
        for _ in range(num_layer):  # 两层 HeteroConv
            # 为每种边类型定义 GCNConv 卷积层
            convs = {
                edge_type: SAGEConv(-1, hidden_dim)  # 使用自动匹配输入维度
                for edge_type in metadata['edge_types']
            }
            # 使用 HeteroConv 聚合不同边类型的卷积操作
            self.convs.append(HeteroConv(convs, aggr='sum'))

            # 定义最后一个线性层
        self.lin = torch.nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x_dict, edge_index_dict):
        # 遍历 x_dict，确保每种节点类型都有自环
        for node_type, x in x_dict.items():
            # 如果该节点类型的自环边类型还未在 edge_index_dict 中，则添加
            self_loop_edge_type = (node_type, 'self_loop', node_type)

            if self_loop_edge_type not in edge_index_dict:
                # 创建自环边索引，使每个节点都连到自身
                self_loop_index = torch.arange(x.size(0)).unsqueeze(0).repeat(2, 1)
                edge_index_dict[self_loop_edge_type] = self_loop_index

        # 逐层应用 HeteroConv
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # 对每种节点类型应用激活函数
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 应用线性层，将节点特征映射到指定的嵌入维度
        return {key: self.lin(x) for key, x in x_dict.items()}


class EdgePredictionDecoder(torch.nn.Module):
    """
    边预测解码器
    基于节点的嵌入向量 z 和填充的 edge_index 来预测边的存在性
    """
    def forward(self, z_dict, edge_index_dict):
        edge_logits_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_logits = (z_dict[src_type][edge_index[0]] * z_dict[dst_type][edge_index[1]]).sum(dim=-1)
            edge_logits_dict[edge_type] = torch.sigmoid(edge_logits)
        return edge_logits_dict


class HeteroGraphAutoencoder(torch.nn.Module):
    """
    图自编码器
    """
    def __init__(self, metadata, hidden_dim, num_layer, embedding_dim):
        super(HeteroGraphAutoencoder, self).__init__()
        self.encoder = HeteroGCNEncoder(metadata, hidden_dim, num_layer, embedding_dim)
        self.decoder = EdgePredictionDecoder()

    def forward(self, data):
        # 通过编码器将节点特征压缩到低维
        z_dict = self.encoder(data.x_dict, data.prior_edge_index_dict)
        # 通过解码器重构边
        reconstructed_edges = self.decoder(z_dict, data.true_edge_index_dict)
        return reconstructed_edges