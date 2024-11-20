import torch
from sklearn.metrics import roc_auc_score
import torch_geometric
from torch_geometric.utils import negative_sampling

def test(model, dataloader, criterion, device,Negative_rate):
    model.eval()  # 设置模型为评估模式
    total_loss = 0

    all_labels = []  # 存储所有的真实标签
    all_predictions = []  # 存储所有的预测概率

    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:
            batch = batch.to(device)

            # 获取编码器生成的节点嵌入
            z_dict = model.encoder(batch.x_dict, batch.prior_edge_index_dict)
            # output = model(batch.x_dict, batch.edge_index_dict)

            # 对每种边类型进行负采样，并构造测试边
            edge_index_dict_test = {}
            edge_labels_dict = {}

            for edge_type, edge_index in batch.true_edge_index_dict.items():
                if edge_index.size(1) == 0:
                    continue
                # 获取正样本边
                pos_edge_index = edge_index.to(device)
                # 生成负采样边

                src, _, dst = edge_type
                if (src != dst):
                    num_nodes = (z_dict[src].size(0), z_dict[dst].size(0))
                else:
                    num_nodes = (z_dict[src].size(0))

                neg_edge_index = negative_sampling(pos_edge_index,
                                                   num_nodes=num_nodes,
                                                   num_neg_samples=pos_edge_index.size(1) * Negative_rate,
                                                   method='sparse')

                # 合并正负样本边并创建标签
                edge_index_dict_test[edge_type] = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                edge_labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))])
                edge_labels_dict[edge_type] = edge_labels.to(device)

            # 解码器重构边
            reconstructed_edges = model.decoder(z_dict, edge_index_dict_test)

            # 计算损失
            loss = 0
            for edge_type, edge_logits in reconstructed_edges.items():
                loss += criterion(edge_logits, edge_labels_dict[edge_type])

                # 收集预测值和真实标签用于后续计算准确性指标
                all_labels.append(edge_labels_dict[edge_type].cpu())
                all_predictions.append(edge_logits.cpu())

            total_loss += loss.item()

    # 计算 AUC 或其他指标
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        auc = float('nan')

    total_loss = total_loss / len(dataloader)


    return total_loss, auc