import numpy as np


def get_ramp_up_value(current_epoch, ramp_up_epochs, initial_weight=0, final_weight=1, mode='gaussian'):
    """
    计算一致性权重的渐进值

    Args:
        current_epoch: 当前epoch
        ramp_up_epochs: 渐进过程的总epoch数
        initial_weight: 初始权重
        final_weight: 最终权重
        mode: 渐进模式，可选'gaussian'或'cosine'

    Returns:
        float: 当前epoch的权重值
    """
    if current_epoch >= ramp_up_epochs:
        return final_weight

    ratio = current_epoch / ramp_up_epochs

    if mode == 'gaussian':
        # 使用高斯函数实现平滑的渐进
        ramp_value = np.exp(-5 * (1 - ratio) ** 2)
    else:  # cosine
        # 使用余弦函数实现平滑的渐进
        ramp_value = (1 - np.cos(np.pi * ratio)) / 2

    return initial_weight + (final_weight - initial_weight) * ramp_value
