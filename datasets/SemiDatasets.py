import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from datasets import split_classes



def get_n_classes(name):
    if name == 'cifar10':
        n_classes = 10
    elif name == 'cifar100':
        n_classes = 100
    elif name == 'fashionmnist':
        n_classes = 10
    elif name == 'svhn':
        n_classes = 10
    elif name == 'mnist':
        n_classes = 10
    else:
        raise NotImplementedError(f'Unknown dataset: {name}')
    return n_classes


def get_dataset_by_name(name, train=True, transform=None):
    if name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    elif name == 'fashionmnist':
        dataset = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    elif name == 'svhn':
        dataset = datasets.SVHN(root='./data', split='train' if train else 'test', download=True, transform=transform)
        data = dataset.data
        targets = dataset.labels
    elif name == 'mnist':
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    else:
        raise NotImplementedError('Dataset {} not implemented'.format(name))
    return data, targets


def get_FedDataset_list(data_name, class_split_num,class_split_edge_num, class_split_edge_id, labeled_ratio=0.1, train=True, transform=None, distributed="nonIID"):
    """
    获取FedSemiCIFAR10数据集
    :param data_name: 数据集名称
    :param class_split_num: 需要将class分为多少个（有多少个端）（分出的类别允许重复）
    :param labeled_ratio: 有标签的数据包含的数量
    :param train: 是否为训练（if False 所有的 labeled_ratio 都为 True）
    :param transform: 数据集的Transform
    :return: 包含了 class_split_num 个 dataset 的列表
    """
    if distributed == "nonIID": # 边缘服务器下每个客户端不同分布
        accept_classes_label = split_classes(get_n_classes(data_name), class_split_num)
        print(accept_classes_label)
        dataset_list = []
        for accept_classes in accept_classes_label:
            dataset = FedSemiDataset(labeled_ratio=labeled_ratio, train=train, transform=transform,
                                     accept_classes=accept_classes, data_name=data_name)
            dataset_list.append(dataset)
    elif distributed == "IID": # 边缘服务器下每个客户端同分布
        dataset_list = []
        for i in range(0,class_split_num):
            dataset = FedSemiDataset(labeled_ratio=labeled_ratio, train=train, transform=transform,
                                     accept_classes=None, data_name=data_name)
            dataset_list.append(dataset)
    else:
        raise ValueError(f"Unknown ditributed: {distributed}")

    return dataset_list


class ImgDataset(Dataset):
    def __init__(self, data_name, train=True, transform=None):
        self.transform = transform
        self.data_name = data_name
        self.data, self.targets = get_dataset_by_name(data_name, train, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if self.data_name == 'svhn':
            image = image.transpose(1, 2, 0)
        if self.data_name == 'fashionmnist':
            image = image[..., np.newaxis]
            image = np.concatenate([image, image, image], axis=-1)
        if self.transform is not None:
            image = self.transform(image)
        return image, target



class FedSemiDataset(Dataset):
    def __init__(self, labeled_ratio, train=True, transform=None, data_name='cifar10', accept_classes=None):
        self.transform = transform
        self.data_name = data_name

        self.weak_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), antialias=True),  # 更大范围的随机裁剪
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # 随机颜色抖动
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

        self.data, self.targets = get_dataset_by_name(data_name, train, transform)
        n_classes = get_n_classes(data_name)

        if accept_classes is not None:
            # 确保所有的accept的值合法
            assert all(0 <= c < n_classes for c in accept_classes)

            # 获取所有数据的标签
            targets = np.array(self.targets)

            # 创建掩码，只保留指定类别
            mask = np.zeros_like(targets, dtype=bool)
            for c in accept_classes:
                mask = mask | (targets == c)    # 位运算操作

            # 过滤数据和标签
            self.data = self.data[mask]
            self.targets = np.array(self.targets)[mask]

        # 设置半监督
        self.labeled_radio = torch.zeros(len(self.data), dtype=torch.bool)
        labeled_indices = torch.randperm(len(self.data))[:int(len(self.data) * labeled_ratio)]
        if train:
            self.labeled_radio[labeled_indices] = True
        else:
            self.labeled_radio[:] = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        is_labeled = self.labeled_radio[index]

        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if len(img.shape) == 2:  # 如果为单通道
            img = img[..., np.newaxis]  # [H, W, C]
            img = np.concatenate([img, img, img], axis=-1)
        if self.data_name == 'svhn':
            img = img.transpose(1, 2, 0)

        if self.transform is not None:
            img = self.transform(img)
        strong_img = self.strong_transform(img)
        weak_img = self.weak_transform(img)

        return {"img": img, "label": label, "is_labeled": is_labeled, "weak_img": weak_img, "strong_img": strong_img}

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FedSemiDataset(labeled_ratio=0.1, train=False, transform=transform)
    print(len(dataset))
    # dataset = FedSemiCIFAR10(labeled_ratio=0.1, train=True, transform=transform, accpet_classes=[0, 3, 5])
    # for i in range(10):
    #     img, label, is_labeled = dataset[i]
    #     print(f"img: {img.shape}, label: {label}, is_labeled: {is_labeled}")
    # print(split_classes(10, split_num=4))
    dataset_list = get_FedDataset_list(class_split_num=4, labeled_ratio=0.1, train=True, transform=transform)
    for dt in dataset_list:
        print(len(dt))
    # dataloader = BatchDataloader(dataset=dataset_list[0], batch_size=16, shuffle=True)
    # img, label, islabeled = dataloader.get_batch()
    # print(f"img: {img.shape}, label: {label}, is_labeled: {islabeled}")
