import datetime
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tree.tree import create_tree
from datasets.SemiDatasets import FedSemiDataset, get_n_classes
from utils.losses import get_loss_function
from torchvision.transforms import transforms
from utils.warmup import get_ramp_up_value
from collections import OrderedDict
from models.encoder import get_encoder
from utils.evaluate import evaluate
import wandb
import random


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch

def laplace_noise(shape, scale, device):
    # 生成均匀分布 [-0.5, 0.5]
    uniform_noise = torch.rand(shape, device=device) - 0.5
    # 拉普拉斯噪声公式：-scale * log(1 - 2|u|)
    raw_noise = -scale * torch.log(1 - 2 * uniform_noise.abs() + 1e-6)  # 加 1e-6 防止 log(0)
    raw_noise = raw_noise * uniform_noise.sign()  # 恢复对称性

    # 将噪声归一化到 [0, 1]
    min_val = raw_noise.min()
    max_val = raw_noise.max()
    noise = (raw_noise - min_val) / (max_val - min_val)

    return noise


def edge_run_loop(cloud_node, config, consistency_weight, classification_criterion, consistency_criterion, device,
                  update=False):
    """
    边的训练循环（一个Batch） 时间复杂度：边数*端数（一个Batch可以在GPU运算？）
    :param device: device
    :param consistency_criterion: 无标签损失
    :param classification_criterion: 有标签损失
    :param update: 是否返回云的权重，即cloud_state_dict_list
    :param consistency_weight: 无监督损失权重
    :param cloud_node: 云节点
    :return: list 所有边的权重列表，用来给云进行EMA更新, loss的记录
    """
    metrics = OrderedDict()
    total_loss = 0
    class_loss = []
    consis_loss = []
    edge_parameters_list = []
    for edge in cloud_node.children:    # 遍历所有的边
        end_output_list = []  # 所有端的输出，用来给边和云输入
        end_labels_list = []
        end_is_labeled_list = [] # 端设备是否有标签
        end_weak_output_list = []
        end_strong_output_list = []
        for end in edge.children:       # 遍历所有的端
            if random.random() < config.models.end_skip_rate:
                continue    # 跳过当前端的数据计算
            batch = end.get_batch_data()    # 获取一个Batch的数据
            end_inputs, end_labels, end_is_labeled = batch["img"], batch["label"], batch["is_labeled"]
            end_inputs = end_inputs.to(device)
            end_labels = end_labels.to(device)
            end_is_labeled = end_is_labeled.to(device)
            end_weak_input = batch['weak_img'].to(device)  # 提供给云
            end_strong_input = batch['strong_img'].to(device)

            # 攻击者
            if random.random() < config.models.attack_rate:
                end_inputs = torch.randn_like(end_inputs, device=device)
                end_labels = torch.randint_like(end_labels, high=get_n_classes(config.datasets.name), device=device)
                end_is_labeled = torch.randint(0, 1, size=end_is_labeled.size(), device=device)
                end_weak_input = torch.randn_like(end_weak_input, device=device)
                end_strong_input = torch.randn_like(end_strong_input, device=device)

            # 运行端模型（编码器）
            end_output = end.run_model(end_inputs)      # 端模型运行，获取特征输出


            if config.train.differential.use:
                a=torch.max(end_output)
                b=torch.min(end_output)
                sensitivity = a-b  # 假设敏感度为 1
                # 计算拉普拉斯噪声尺度
                scale = sensitivity / config.train.differential.epsilon
                # 生成噪声并添加到原始张量
                noise = laplace_noise(end_output.shape, scale, device)
                end_output += noise
                end_output = torch.clamp(end_output, min=b, max=a)
            end_output_list.append(end_output)      # 将特征附加在列表中，等待传输给边
            end_labels_list.append(end_labels)      # 将标签附加在列表中
            end_is_labeled_list.append(end_is_labeled)  # 将标签标记附加在列表中

            # 运行弱增强数据，增强可以对数据进行扩充，鲁棒性更强
            end_weak_output = end.run_model(end_weak_input)
            if config.train.differential.use:
                a = torch.max(end_weak_output)
                b = torch.min(end_weak_output)
                sensitivity_weak = a - b
                # 计算拉普拉斯噪声尺度
                scale_weak = sensitivity_weak / config.train.differential.epsilon
                # 为每个值添加噪声
                noise = laplace_noise(end_weak_output.shape, scale_weak, device)
                end_weak_output += noise
                end_weak_output = torch.clamp(end_weak_output, min=b, max=a)
            end_weak_output_list.append(end_weak_output)

            # 运行强增强数据
            end_strong_output = end.run_model(end_strong_input)
            if config.train.differential.use:
                a = torch.max(end_strong_output)
                b = torch.min(end_strong_output)
                sensitivity_strong = a - b
                # 计算拉普拉斯噪声尺度
                scale_strong = sensitivity_strong / config.train.differential.epsilon
                # 为每个值添加噪声
                noise = laplace_noise(end_strong_output.shape, scale_strong, device)
                end_strong_output += noise
                end_strong_output = torch.clamp(end_strong_output, min=b, max=a)
            end_strong_output_list.append(end_strong_output)

        encoded_inputs = torch.cat(end_output_list, dim=0)  # 边模型的输入，编码后的特征embedding
        del end_output_list
        all_end_labels = torch.cat(end_labels_list, dim=0)  # 所有端的标签（用来计算损失）
        del end_labels_list
        all_end_is_labeled = torch.cat(end_is_labeled_list, dim=0)  # 所有端的是否存在标签标记（用来计算无监督损失）
        del end_is_labeled_list
        all_end_weak = torch.cat(end_weak_output_list, dim=0)
        del end_weak_output_list
        all_end_strong = torch.cat(end_strong_output_list, dim=0)
        del end_strong_output_list
        # torch.cuda.empty_cache()
        edge_classification_output = edge.run_model(encoded_inputs)  # 边运行，运行有标签数据(无增强)

        # 计算损失，保障至少有一个端有标签
        if all_end_is_labeled.sum() > 0:
            classification_loss = classification_criterion(edge_classification_output[all_end_is_labeled],
                                                           all_end_labels[all_end_is_labeled])
        else:
            classification_loss = 0

        # 分类损失计算完毕，节约显存
        del encoded_inputs, all_end_labels, all_end_is_labeled, edge_classification_output
        # torch.cuda.empty_cache()

        # 无论有标签还是无标签，都进行无监督（一致性损失）
        edge_consistency_output = edge.run_model(all_end_strong)
        cloud_consistency_output = cloud_node.run_model(all_end_weak)

        consistency_loss = consistency_criterion(
            F.softmax(edge_consistency_output, dim=1),
            F.softmax(cloud_consistency_output, dim=1)
        )

        # 一致性损失计算完毕，节约显存
        del all_end_strong, all_end_weak, edge_consistency_output, cloud_consistency_output
        # torch.cuda.empty_cache()

        loss = consistency_weight * consistency_loss + classification_loss
        edge.optimizer_step(loss)  # 更新梯度

        total_loss += loss.item()
        # 判断是不是张量决定怎么存储
        class_loss.append(
            classification_loss.item() if isinstance(classification_loss, torch.Tensor) else classification_loss)
        consis_loss.append(consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss)

        if update:  # 将边的权重传递给云
            edge_parameters_list.append(edge.get_parameters())

    metrics['total_loss'] = total_loss / len(cloud_node.children)
    metrics['avg_class_loss'] = sum(class_loss) / len(cloud_node.children)
    metrics['avg_consis_loss'] = sum(consis_loss) / len(cloud_node.children)
    metrics['classification_loss'] = class_loss
    metrics['consistency_loss'] = consis_loss

    if len(edge_parameters_list) > 0:
        # 将所有边的权重求平均后返回
        avg_parameters = []
        for param_index in range(len(edge_parameters_list[0])):
            params = [params[param_index] for params in edge_parameters_list]
            avg_param = torch.stack(params).mean(dim=0)
            avg_parameters.append(avg_param)
        return avg_parameters, metrics
    else:
        return None, metrics

def train(config):
    device = torch.device(f"cuda:{config.train.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config.train.device)

    # 测试用
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize(0.5, 0.5)
    ])

    valset = FedSemiDataset(labeled_ratio=1, train=False, transform=transform, data_name=config.datasets.name)
    valloader = DataLoader(valset, batch_size=config.datasets.batch_size, shuffle=False, num_workers=config.datasets.num_workers)
    test_encoder = get_encoder(config).to(device)

    # 获取损失函数
    classification_criterion, consistency_criterion = get_loss_function(config.train.class_fn, config.train.consis_fn, num_class=get_n_classes(config.datasets.name))

    tree = create_tree(config.models.tree_list, config)
    tree.move_to_device(device)

    current_steps = 0  # 当前运行的步数（Batch）
    beat_accuracy = 0
    while current_steps < config.train.total_steps:
        consistency_weight = get_ramp_up_value(
            current_epoch=current_steps,
            ramp_up_epochs=config.train.ramp_up_steps,
            initial_weight=config.train.initial_weight,
            final_weight=config.train.final_weight,
            mode=config.train.warm_mode
        )
        wandb.log({"consistency_weight": consistency_weight}, step=current_steps)
        avg_parameters, train_metrics = edge_run_loop(cloud_node=tree.root,
                                                      config=config,
                                                      consistency_weight=consistency_weight,
                                                      classification_criterion=classification_criterion,
                                                      consistency_criterion=consistency_criterion,
                                                      device=device,
                                                      update=current_steps  % config.train.ema_update_step == 0
                                                      )
        if avg_parameters is not None:
            tree.root.model.update_by_parameters(avg_parameters)
            wandb.log({"ema_decay": tree.root.model.decay}, step=current_steps)

        wandb.log(train_metrics, step=current_steps)
        if (current_steps + 1) % config.train.log_step == 0:
            # 格式化打印信息
            print(f"\nStep [{current_steps + 1}/{config.train.total_steps}]")
            print("Training Metrics:")
            print(f"├── Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"├── Classification Loss: {train_metrics['avg_class_loss']:.4f}")
            print(f"    ├── Classification Details: {train_metrics['classification_loss']}")
            print(f"└── Consistency Loss: {train_metrics['avg_consis_loss']:.4f}")
            print(f"    ├── Consistency Details: {train_metrics['consistency_loss']}")

        if (current_steps + 1) % config.train.evaluate_step == 0:
            val_metrics = evaluate(
                encoded_model=test_encoder,
                model=tree.root.model,
                val_loader=valloader,
                criterion=classification_criterion,
                device=device
            )
            print("\nValidation Metrics:")
            print(f"├── Loss: {val_metrics['val_loss']:.4f}")
            print(f"└── Accuracy: {val_metrics['val_accuracy']:.2f}%")  # 添加百分号
            wandb.log(val_metrics, step=current_steps)

            # 保存最佳模型
            if val_metrics['val_accuracy'] > beat_accuracy:
                beat_accuracy = val_metrics['val_accuracy']
                os.makedirs(os.path.join(config.train.ckpt_save_path, config.datasets.name), exist_ok=True)
                tree.root.model.save(os.path.join(config.train.ckpt_save_path, config.datasets.name,
                                                  config.train.ckpt_save_name))

        current_steps += 1


if __name__ == '__main__':
    config = OmegaConf.load('config/formated_config.yaml')
    os.makedirs('./logs', exist_ok=True)
    wandb.init(project='TreeFedSemi', dir="logs",
               name=f"FedSemi-{config.datasets.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
               config=OmegaConf.to_container(config), job_type='train')
    train(config)
