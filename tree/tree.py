from typing import Union
import torch
from models.GetModel import get_model
from omegaconf import OmegaConf, DictConfig, ListConfig
from datasets.SemiDatasets import get_FedDataset_list
from datasets import BatchDataloader
from torchvision.transforms import transforms
import torch.optim as optim


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.level = 0  # 0: Cloud, 1: Edge, 2: End
        self.model = None
        self.dataloader = None
        self.dataloader_iterator = None
        self.optimizer = None

    def add_child(self, child_node):
        """添加子节点"""
        if self.level >= 2:
            raise ValueError("云边端树最多只有三层")

        child_node.level = self.level + 1
        self.children.append(child_node)
        return child_node

    def remove_child(self, name):
        """根据值删除子节点"""
        for child in self.children[:]:
            if child.name == name:
                self.children.remove(child)
                return True
        return False

    def run_model(self, *args, **kwargs):
        if self.model is None:
            raise ValueError("Model is None, call self.init_model() first")
        if self.level == 0 or self.level == 2:
            with torch.no_grad():
                return self.model(*args, **kwargs)
        return self.model(*args, **kwargs)

    def init_model(self, config):
        print(f"节点{self.name}模型初始化完毕")
        self.model = get_model(self.level, config)

    def set_dataloader(self, dataloader):
        assert self.level == 2, "只有端才可以设置Dataloader"
        if self.dataloader is None:
            print(f"端{self.name}的Dataloader初始化完毕")
            self.dataloader = dataloader
        else:
            raise ValueError("当前的Dataloader不为空")

    def get_batch_data(self):
        if self.dataloader is None:
            raise ValueError("当前的Dataloader为空")
        return self.dataloader.get_batch()

    def set_optimizer(self, optimizer):
        assert self.level == 1, "只有边才可以设置Optimizer"
        if self.optimizer is None:
            print(f"边{self.name}的Optimizer初始化完毕")
            self.optimizer = optimizer
        else:
            raise ValueError("当前Optimizer不为空")

    def optimizer_step(self, loss):
        if self.optimizer is None:
            raise ValueError("当前的Optimizer为空")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_parameters(self):
        assert self.level == 1, "只有边需要传递patameters"
        if self.model is None:
            raise ValueError("当前的Model为空")
        return [param.data.clone() for param in self.model.parameters()]


class Tree:
    def __init__(self, root_value):
        self.root = TreeNode(root_value)

    def find_node(self, name, node=None):
        """查找具有特定值的节点"""
        if node is None:
            node = self.root

        if node.name == name:
            return node

        for child in node.children:
            result = self.find_node(name, child)
            if result:
                return result
        return None

    def add_node(self, parent_value, child_value):
        """在特定父节点下添加子节点"""
        parent = self.find_node(parent_value)
        if parent is None:
            raise ValueError(f"未找到指定节点: {parent_value}")

        new_node = TreeNode(child_value)
        return parent.add_child(new_node)

    def print_tree(self, node=None, level=0):
        """打印树结构"""
        if node is None:
            node = self.root

        indent = "\t" * level
        print(f"{indent}{node.name}_{node.model.__class__.__name__}")

        for child in node.children:
            self.print_tree(child, level + 1)

    def init_node_model(self, config, node=None):
        """
        使用config中的配置初始化节点node的模型
        :param config: 模型配置
        :param node: 起始节点，默认为None或者root
        :return: None
        """
        if node is None:
            node = self.root

        if node.model is None:
            node.init_model(config)

        for child in node.children:
            self.init_node_model(config, child)

    def move_to_device(self, device, node=None):
        """
        将模型移动到device中
        :param device: Pytorch设备
        :param node: 起始节点，默认为None
        :return: None
        """
        if node is None:
            node = self.root
        if node.model is None:
            raise ValueError(f"在移动模型前需要先初始化，请调用init_node_model()")
        else:
            node.model.to(device)
        for child in node.children:
            self.move_to_device(device, child)

    def init_end_dataloader(self, config):
        print("开始初始化端的Dataloader".center(80, "*"))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(0.5, 0.5)
        ])
        edge_id=0
        for edge in self.root.children:
            end_num = len(edge.children)  # 获取所有端的数量，用来生成数据集
            edge_num = len(self.root.children)  # 获取所有端的数量，用来生成数据集
            dataset_list = get_FedDataset_list(class_split_num=end_num,
                                               class_split_edge_num=edge_num,
                                               class_split_edge_id=edge_id,
                                                   labeled_ratio=config.datasets.labeled_ratio,
                                                   train=True,
                                                   transform=transform,
                                                   data_name=config.datasets.name,
                                                   distributed=config.datasets.distributed)
            edge_id += 1
            assert len(dataset_list) == end_num, f"Dataset的数量必须要和端的数量相等，当前dataset:{len(dataset_list)},end_num: {end_num}"
            for end, dataset in zip(edge.children, dataset_list):
                dataloader = BatchDataloader(dataset, batch_size=config.datasets.batch_size, shuffle=True,
                                             num_workers=config.datasets.num_workers, pin_memory=config.datasets.pin_memory,
                                             persistent_workers=config.datasets.persistent_workers,
                                             prefetch_factor=config.datasets.prefetch_factor)
                # dataloader = DataLoader(dataset, batch_size=config.datasets.batch_size, shuffle=True,
                #                         num_workers=config.datasets.num_workers, pin_memory=config.datasets.pin_memory,
                #                         persistent_workers=config.datasets.persistent_workers)
                end.set_dataloader(dataloader)
        print("初始化端的Dataloader完毕".center(80, "*"))

    def init_edge_optimizer(self, config):
        print("开始初始化边的Optimizer".center(80, "*"))
        for edge in self.root.children:
            if edge.model is None:
                raise ValueError("设置Optimizer之前必须先初始化Model")
            if config.train.optimizer == "AdamW":
                optimizer = optim.AdamW(edge.model.parameters(), lr=config.train.learning_rate)
                edge.set_optimizer(optimizer)
        print("初始化边的Optimizer完毕".center(80, "*"))


def create_tree(create_list: list, config: Union[DictConfig, ListConfig]) -> Tree:
    """
    创建树结构
    :param config: 模型的配置
    :param create_list: 树结构列表，例：[3, 2, 1] 即表示为 1个云 3个边[A, B, C] 边内部包含[A1, A2, A3][B1, B2][C1]端
    :return: 根节点（云）
    """
    cloud_tree = Tree("Cloud")
    for edge in range(len(create_list)):
        edge_name = f"Edge{edge}"
        cloud_tree.add_node("Cloud", edge_name)
        for end in range(create_list[edge]):
            end_name = f"{edge_name}_End{end}"
            cloud_tree.add_node(edge_name, end_name)
    print("开始初始化节点模型".center(80, "*"))
    cloud_tree.init_node_model(config)
    print("初始化节点模型完毕".center(80, "*"))
    cloud_tree.init_end_dataloader(config)
    cloud_tree.init_edge_optimizer(config)
    print("开始训练".center(80, "*"))
    return cloud_tree


if __name__ == '__main__':
    config = OmegaConf.load('config/formated_config.yaml')
    tree = create_tree(config.models.tree_list, config)
    # tree.print_tree()
    print("验证端模型是否为同一个")
    print(tree.root.children[0].children[0].model is tree.root.children[0].children[1].model)
    print("Finish")
