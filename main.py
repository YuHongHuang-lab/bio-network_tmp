import yaml
import logging
import argparse
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from dataset.bio_dataset import BioDataset
from model.GraphAutoencoder import HeteroGraphAutoencoder
from train import train
from test import test


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path="config.yaml"):
    ####################
    # 配置与日志设置数据集 #
    ####################
    config = load_config(config_path)

    # 配置日志格式和级别
    if config["log"]["level"]:
        if config["log"]["level"] == "INFO":
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        elif config["log"]["level"] == "DEBUG":
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        # Todo
        # 还有其他日志级别
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(config)

    ####################
    #     生成数据集     #
    ####################
    dataset = BioDataset(config=config["dataset"])
    # 输出数据集信息
    logging.info(f'数据集元数据: {dataset.get_metadata()}')
    logging.info(f'子图数量: {len(dataset)}')
    logging.info(f'第一个图的数据: {dataset[0]}')


    ####################
    #     配置加载器     #
    ####################
    # loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=config["training"]["batch_size"])
    train_data_list = []
    test_data_list = []

    # 从 dataset 中提取边数据

    prior_edge_index_dict = {edge_type: dataset.data[edge_type].edge_index for edge_type in dataset.data.edge_types}
    prior_edge_attr_dict = {edge_type: dataset.data[edge_type].edge_attr for edge_type in dataset.data.edge_types}

    test_edge_types = [edge_type for edge_type in dataset.data.edge_types if hasattr(dataset.data[edge_type], 'test_edge_index')]

    train_data_list.append(Data(
        x_dict=dataset.data.x_dict,
        prior_edge_index_dict=prior_edge_index_dict,
        prior_edge_attr_dict=prior_edge_attr_dict,
        true_edge_index_dict={
            edge_type: dataset.data[edge_type].train_edge_index[:, dataset.data[edge_type].train_edge_attr[:, -1] == 1] for
            edge_type in dataset.data.edge_types},
        true_edge_attr_dict={edge_type: dataset.data[edge_type].train_edge_attr[dataset.data[edge_type].train_edge_attr[:, -1] == 1]
                             for edge_type in dataset.data.edge_types}
    ))



    test_data_list.append(Data(
        x_dict=dataset.data.x_dict,
        prior_edge_index_dict=prior_edge_index_dict,
        prior_edge_attr_dict=prior_edge_attr_dict,
        true_edge_index_dict={
            edge_type: dataset.data[edge_type].test_edge_index[:, dataset.data[edge_type].test_edge_attr[:, -1] == 1] for
            edge_type in test_edge_types},
        true_edge_attr_dict={edge_type: dataset.data[edge_type].test_edge_attr[dataset.data[edge_type].test_edge_attr[:, -1] == 1]
                             for edge_type in test_edge_types}
    ))

    train_loader = DataLoader(train_data_list, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=config['training']['batch_size'])


    ####################
    #     模型实例化     #
    ####################
    model = HeteroGraphAutoencoder(metadata=dataset.get_metadata(),
                                   hidden_dim=config['model']['hidden_dim'],
                                   num_layer=config['model']['num_layer'],
                                   embedding_dim=config['model']['num_layer'])

    ####################
    #    损失函数实例化   #
    ####################
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()

    ####################
    #     优化器实例化   #
    ####################
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    ####################
    #     训练与测试     #
    ####################
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_accuracy = 0
    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        train_loss = train(model, train_loader,
                           optimizer=optimizer,
                           criterion=criterion,
                           device=device,
                           Negative_rate=config['model']['Negative_rate'])
        test_loss, test_auc = test(model, test_loader,
                                   criterion=criterion,
                                   device=device,
                                   Negative_rate=config['model']['Negative_rate'])

        logging.info(f"Epoch {epoch+1}/{epochs},"
                     f" Train Loss: {train_loss:.4f}, "
                     f"Test Loss: {test_loss:.4f},"
                     f"Test AUC: {test_auc:.4f}")

        # 保存最佳模型
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     torch.save(model.state_dict(), config["training"]["model_save_path"])
        #     print(f"Best model saved with accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BioNetwork机器学习框架")
    parser.add_argument('--config', type=str, default='config.yaml', help="运行参数")
    main(parser.parse_args().config)
    # main(config_path='config.yaml')
