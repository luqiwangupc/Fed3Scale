import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np


class BatchDataloader:
    """带计数器的批次数据加载器类，支持多进程数据加载"""
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, worker_init_fn=None,
                 prefetch_factor=2, persistent_workers=False):
        """
        初始化加载器
        Args:
            dataset: PyTorch数据集对象
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的工作进程数
            pin_memory: 是否将数据放入固定内存中（GPU训练时建议True）
            worker_init_fn: 工作进程初始化函数
            prefetch_factor: 预加载因子，每个工作进程预加载的批次数
            persistent_workers: 在数据集遍历完后是否保持工作进程存活
        """
        # 设置多进程启动方法
        if num_workers > 0:
            try:
                mp.set_start_method('fork')
            except RuntimeError:
                pass    # 已经设置过启动方法则忽略

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else None,
        )
        self.iterator = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._init_iterator()

    def _init_iterator(self):
        """初始化迭代器"""
        self.iterator = iter(self.dataloader)

    def get_batch(self):
        if self.iterator is None:
            self._init_iterator()

        try:
            batch = next(self.iterator)
        except StopIteration:
            self._init_iterator()
            batch = next(self.iterator)

        return batch

    def __del__(self):
        """析构函数，确保正确清理资源"""
        if hasattr(self, 'dataloader'):
            del self.dataloader
            del self.iterator

def split_classes(max_length, split_num, edge_id=-1):
    """
    将0~max_length的数字分配到split_num个尽可能长的数组中
    :param max_length: 最大的值（类别数量）
    :param split_num: 需要拆分的数量
    :return: 拆分后的数组
    """
    if split_num <= 0:
        raise ValueError(f"split_num 必须大于 0， 当前为： {split_num}")
    if edge_id != -1:
        numbers = np.random.permutation(max_length)+edge_id
    else:
        numbers = np.random.permutation(max_length)
    result = []

    if split_num <= max_length:
        base_length = max_length // split_num
        extra_count = max_length % split_num

        # 计算每个数组的起始和结束索引
        start_indices = np.zeros(split_num, dtype=int)
        lengths = np.full(split_num, base_length)
        lengths[:extra_count] += 1  # 前extra_count个数组多分配一个数字
        end_indices = np.cumsum(lengths)
        start_indices[1:] = end_indices[:-1]

        # 分割数组
        result = [numbers[start:end] for start, end in zip(start_indices, end_indices)]

    else:
        # 当n>max_length时，允许数字重复
        base_length = max(1, 10 // split_num)  # 确保至少长度为1

        # 使用roll函数创建循环移动的数组
        for i in range(split_num):
            # 循环移动数组并取前base_length个元素
            shifted_array = np.roll(numbers, -i)[:base_length]
            result.append(shifted_array)

    return result
