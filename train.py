import torch
from torch_geometric.utils import negative_sampling


def train(model, dataloader, criterion, optimizer, device, Negative_rate):
    model.train()

    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 获取编码器生成的节点嵌入
        z_dict = model.encoder(batch.x_dict, batch.prior_edge_index_dict)
        # reconstructed_edges = model(batch)

        # 对每种边类型进行负采样，并构造训练边
        edge_index_dict_train = {}
        edge_labels_dict = {}

        for edge_type, edge_index in batch.true_edge_index_dict.items():
            if edge_index.size(1)==0:
                continue
            # 获取正样本边
            pos_edge_index = edge_index.to(device)
            # 生成负样本边
            src, _, dst = edge_type
            if (src!=dst):
                num_nodes = (z_dict[src].size(0), z_dict[dst].size(0))
            else:
                num_nodes = (z_dict[src].size(0))
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_nodes=num_nodes,
                                               num_neg_samples=pos_edge_index.size(1)*Negative_rate,
                                               method='sparse')

            # 将正负样本边合并，并创建标签
            edge_index_dict_train[edge_type] = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))])
            edge_labels_dict[edge_type] = edge_labels.to(device)

        # 通过解码器重构边
        reconstructed_edges = model.decoder(z_dict, edge_index_dict_train)

        # 计算损失
        loss = 0
        for edge_type, edge_logits in reconstructed_edges.items():
            loss += criterion(edge_logits, edge_labels_dict[edge_type])

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
