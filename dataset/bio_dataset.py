import os
from importlib.metadata import metadata
from tqdm import tqdm
from collections import Counter

import yaml
import torch
import logging
import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from torch_geometric.data import InMemoryDataset, HeteroData, DataLoader


class BioDataset(InMemoryDataset):
    def __init__(self, config):
        self.config = config

        ################################
        #    从配置文件中获取路径和参数    #
        ################################
        # 数据集根目录
        self.data_root = self.config['data_root']
        # 子图数量
        self.num_graphs = self.config['num_graphs']
        # 是否使用节点名称
        self.use_node_name = self.config['use_node_name']
        # 是否读取图标签
        self.use_graph_labels = self.config.get('use_graph_labels', False)

        # 节点label列名字
        self.node_label_column = self.config.get('node_label_column', None)
        # 边时间戳列名
        self.edge_timestamp_column = self.config.get('edge_timestamp_column', None)
        # 边类型列名
        self.edge_type_column = self.config.get('edge_type_column', None)
        # 边label列名
        self.edge_label_column = self.config.get('edge_label_column', None)

        # 是否合并图
        self.merge_graphs = self.config.get('merge_graphs', False)

        self.transform = None
        self.pre_transform = None

        # 调用父类构造函数
        super(BioDataset, self).__init__(self.data_root, self.transform, self.pre_transform, force_reload=self.config["force_reload"])
        self.data, self.slices = self._load_processed_data()

    def _load_processed_data(self):
        """
        加载处理后的数据
        :return:
        """
        if self.merge_graphs:
            # 合并子图
            loaded_data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]), weights_only=False)
            if isinstance(loaded_data, tuple):
                return loaded_data  # 返回 (dataset, slices)
            else:
                return loaded_data, None  # 单张子图的情况
        else:
            # 非合并模式
            # 由 __getitem__ 控制加载
            return None, None

    @property
    def raw_dir(self) -> str:
        """
        读取配置设置原始文件目录
        :return:
        """
        return self.config['raw_dir']

    @property
    def raw_file_names(self):
        """
        读取配置获取原始文件列表
        :return:
        """
        raw_file_list = []
        for i in range(self.num_graphs):
            if os.path.exists(os.path.join(self.raw_dir, f"graph_{i}_nodes.csv")):
                raw_file_list.append(f"graph_{i}_nodes.csv")
            if os.path.exists(os.path.join(self.raw_dir, f"graph_{i}_edges.csv")):
                raw_file_list.append(f"graph_{i}_edges.csv")
            if os.path.exists(os.path.join(self.raw_dir, f"graph_{i}_node_names.csv")):
                raw_file_list.append(f"graph_{i}_node_names.csv")

        return raw_file_list

    @property
    def processed_dir(self) -> str:
        """
        读取配置设置处理后文件目录
        :return:
        """
        return self.config['processed_dir']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """
        生成处理后文件名列表
        :return:
        """
        if self.merge_graphs:
            return ['merged_data.pt']
        else:
            return [f'graph_{i}.pt' for i in range(self.num_graphs)
            ]

    @property
    def processed_paths(self) -> List[str]:
        """返回处理后的文件路径。"""
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def download(self) -> None:
        pass

    def process(self):

        seed = self.config.get('seed', 42)
        rate = self.config.get('rate', 0.2)

        data_list = []

        # 如果使用标签，从标签文件读取标签
        if self.use_graph_labels:
            labels_df = pd.read_csv(os.path.join(self.raw_dir, 'graph_labels.csv'))

        for i in range(self.num_graphs):
            # 从自定义原始数据目录读取 CSV 文件
            nodes_file = os.path.join(self.raw_dir, f'graph_{i}_nodes.csv')
            edges_file = os.path.join(self.raw_dir, f'graph_{i}_edges.csv')

            # 读取节点和边的原始数据
            nodes = pd.read_csv(nodes_file)
            edges = pd.read_csv(edges_file)

            # 判断是否使用节点名称索引各个节点
            if self.use_node_name:
                # 使用节点名称

                # 读入节点名称
                node_names_df = pd.read_csv(os.path.join(self.raw_dir, f'graph_{i}_node_names.csv'))
                # 初始化节点名称字典
                node_dict = dict()
                # 初始化节点id字典
                id_dict = dict()
                # 遍历节点名称
                for index, row in node_names_df.iterrows():
                    # 判断当前节点类型是否第一次出现
                    if row.get("type", None) in id_dict:
                        # 非第一次出现则对应类型节点的id递增1
                        id = id_dict[row.get("type", None)] + 1
                    else:
                        # 第一次出现该类型节点id为0
                        id = 0
                    id_dict[row.get("type", None)] = id

                    # 写入节点名称字典
                    node_dict[row["name"]] = {
                        "id": id,
                        "type": row.get("type", None)
                    }

                # 将节点id和类型写入节点原始数据
                nodes['_id'] = nodes.iloc[:, 0].map(lambda x: node_dict[x]["id"])
                nodes['_type'] = nodes.iloc[:, 0].map(lambda x: node_dict[x]["type"])

                # 创建 HeteroData 对象，根据情况包含标签
                data = HeteroData()

                # 按节点类型对节点原始数据分组
                # 遍历全部分组
                grouped = nodes.groupby('_type')
                for type_name, group in grouped:
                    # 删除节点类型
                    group = group.drop(columns=['_type'])
                    # 按节点id排序
                    group = group.sort_values(by='_id')
                    # 删除节点id
                    group = group.drop(columns=['_id'])

                    # 判断是否设置节点label列
                    if self.node_label_column:
                        # 若存在节点label则提取后删除该列
                        data[str(type_name)].y = torch.tensor(group[self.node_label_column], dtype=torch.float)
                        group = group.drop(columns=[self.node_label_column])

                    # 提取其余列作为节点特征
                    x = torch.tensor(group.iloc[:, 1:].values, dtype=torch.float)
                    data[str(type_name)].x = x

                # 提取边索引和边特征
                edge_dict = dict()

                # 初始化边类型列为None
                edge_type_column = None
                # 判断是否设置边类型列
                if self.edge_type_column:
                    # 若设置则提取并删除边原始数据中的边类型列
                    edge_type_column = self.edge_type_column
                    edges = edges.drop(columns=[edge_type_column])
                # num = 0
                for index, row in tqdm(edges.iterrows(), total=len(edges), desc='Processing edges'):
                    # num+=1
                    # print(num)
                    # 前两列确定为边的src和dst
                    # 判断src和dst对应的节点名称是否在节点名称字典中存在
                    if row.iloc[0] in node_dict and row.iloc[1] in node_dict:
                        # 是否存在边类型列
                        if edge_type_column:
                            # 存在
                            edge_type = (node_dict[row.iloc[0]]["type"], row[edge_type_column], node_dict[row.iloc[1]]["type"])
                        else:
                            # 不存在
                            edge_type = (node_dict[row.iloc[0]]["type"], 'edge', str(node_dict[row.iloc[1]]["type"]))

                        if edge_type not in edge_dict:
                            edge_dict[edge_type] = {
                                "index": [],
                                "attr": []
                            }

                        edge_dict[edge_type]["index"].append([node_dict[row.iloc[0]]["id"], node_dict[row.iloc[1]]["id"]])
                        if len(row) > 2:
                            edge_dict[edge_type]["attr"].append(row[2:].values)

                train_edges, test_edges = self.split_edges(edge_dict, rate=rate, seed=seed)

                # 遍历生成好的边字典填充HetroData对象
                for key in edge_dict:
                    # 获取边索引并转置为需要的形状
                    edge_index = np.array(edge_dict[key]["index"]).T
                    # 设置遍索引
                    data[key].edge_index = torch.tensor(edge_index, dtype=torch.long)
                    # 判断是否存在边特征
                    if len(edge_dict[key]["attr"]) > 0:
                        # 存在则设置边特征
                        data[key].edge_attr = torch.tensor(np.array(edge_dict[key]["attr"],dtype=float), dtype=torch.float)

                # 添加训练集边
                for key, edges in train_edges.items():
                    edge_index = np.array(edges["index"]).T
                    data[key].train_edge_index = torch.tensor(edge_index, dtype=torch.long)
                    data[key].train_edge_attr = torch.tensor(np.array(edges["attr"], dtype=float),
                                                             dtype=torch.float)

                # 添加测试集边
                for key, edges in test_edges.items():
                    edge_index = np.array(edges["index"]).T
                    data[key].test_edge_index = torch.tensor(edge_index, dtype=torch.long)
                    data[key].test_edge_attr = torch.tensor(np.array(edges["attr"], dtype=float),
                                                            dtype=torch.float)

            else:
                # 不使用节点名称
                # Todo
                # 待实现
                pass

            # 获取图的标签（如果使用标签）
            if self.use_graph_labels:
                data["global"].y = torch.tensor([labels_df.loc[labels_df['graph_id'] == i, 'label'].values[0]], dtype=torch.long)

            data_list.append(data)

        # 确保处理目录存在
        os.makedirs(self.processed_dir, exist_ok=True)

        if self.merge_graphs:
            if len(data_list) > 0:
                # 合并所有 Data 对象并保存为一个文件
                if len(data_list) == 1:
                    # 只有一张子图的情况，不使用 collate()
                    torch.save(data_list[0], os.path.join(self.processed_dir, self.processed_file_names[0]))
                    logging.info(f"单张子图已保存至 {self.processed_file_names[0]}")
                else:
                    data, slices = self.collate(data_list)
                    torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[0]))
                    logging.info(f"所有图数据已合并并保存至 {self.processed_file_names[0]}")
            else:
                raise Exception("The dataset is empty.")
        else:
            # 分别保存每个 Data 对象
            for i, data in enumerate(data_list):
                torch.save((data, None), self.processed_file_names[i])
                logging.info(f"图数据 {i} 已保存至 {self.processed_file_names[i]}")

    def __getitem__(self, idx):
        """
        根据配置加载处理后文件
        :param idx:
        :return:
        """
        if self.merge_graphs:
            if self.slices is None:
                # 只有一张子图，直接返回
                return self._data
            else:
                # 多张子图，使用切片信息提取
                return self.get(idx)
        else:
            data, _ = torch.load(self.processed_file_names[idx])
            return data

    def __len__(self):
        """
        返回数据集的大小。
        :return:
        """
        if self.merge_graphs:
            # 从合并后的文件中加载数据和切片信息
            if self.slices is None:
                return 1  # 单张子图的情况
            else:
                # 返回合并后图的数量，通过 slices['x'] 的长度 - 1 确定
                return len(self.slices['x']) - 1
        else:
            # 如果没有合并，直接返回图的数量
            return self.num_graphs

    def get(self, idx):
        """根据索引从合并的数据中提取子图。"""
        data = HeteroData()

        # 遍历每个属性，并根据切片信息提取数据
        for key in self._data.keys():
            item = self._data[key]
            if torch.is_tensor(item):
                start, end = self.slices[key][idx], self.slices[key][idx + 1]
                data[key] = item[start:end]
            else:
                data[key] = item

        return data

    def get_metadata(self):
        dataset_metadata = {
            'node_types': self.data.node_types,
            'edge_types': self.data.edge_types,
        }

        return dataset_metadata


    def split_edges(self, edge_dict, rate=0.2, seed=42):
        np.random.seed(seed)
        train_edges = {}
        test_edges = {}

        for edge_type, edge_data in edge_dict.items():
            edge_attrs = edge_data['attr']
            edge_indices = edge_data['index']

            mask = [attr[-1] == 1 for attr in edge_attrs]  # 假设 edge_id_T 是 attr 的最后一列
            true_edges = [edge for edge, is_true in zip(edge_indices, mask) if is_true]

            print('_'.join(edge_type)+': true edges: '+str(Counter(mask)[1])+'\t'+'prior edges: '+str(Counter(mask)[0]))

            true_attrs = [attr for attr, is_true in zip(edge_attrs, mask) if is_true]

            if len(true_edges) == 0:
                # 如果该边类型没有 edge_id_T=1 的边，全部边作为训练集
                train_edges[edge_type] = {
                    "index": edge_indices,
                    "attr": edge_attrs,
                }
                continue

            # 对 edge_id_T=1 的边进行随机切分
            indices = list(range(len(true_edges)))
            np.random.shuffle(indices)  # 使用固定种子随机打乱
            split_point = int(len(indices) * (1 - rate))
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]

            # 构造训练集和测试集
            train_edges[edge_type] = {
                "index": [true_edges[i] for i in train_indices],
                "attr": [true_attrs[i] for i in train_indices],
            }
            test_edges[edge_type] = {
                "index": [true_edges[i] for i in test_indices],
                "attr": [true_attrs[i] for i in test_indices],
            }

        return train_edges, test_edges
