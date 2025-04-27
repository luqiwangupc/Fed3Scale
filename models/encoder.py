from typing import Union
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, VGG16_Weights, MobileNet_V3_Small_Weights, EfficientNet_V2_S_Weights, EfficientNet_B2_Weights, ResNet34_Weights
from utils.singleton import singleton
from collections import OrderedDict


def get_mae_encoder(name, in_channels=3, ckpt_path=None):
    print(f"使用MAE预训练模型{name}, in_channels={in_channels}")
    if ckpt_path is not None:   # 读取权重，由于原始保存的权重是pytorch_lightning类型，需要转化
        state_dict = OrderedDict()
        pl_state_dict = torch.load(ckpt_path)['state_dict']
        for k, v in pl_state_dict.items():
            if k.startswith('model.encoder'):
                state_dict[k.replace('model.encoder.', '')] = pl_state_dict[k]
        model.load_state_dict(state_dict)
        # model.initialize()
        # model.eval()
    return model

def get_encoder(config: Union[str, DictConfig]):
    """重构函数，为了兼容原始的str和现在新的config"""
    if isinstance(config, DictConfig):
        name = config.models.encoder_name
        if config.models.pretrain_ckpt is not None:     # 使用预训练encoder
            in_channels = 1 if config.datasets.name == 'fashionmnist' else 3
            model = get_mae_encoder(name, in_channels, config.models.pretrain_ckpt)
        else:
            model = get_encoder_by_name(name)
        return model
    if isinstance(config, str):     # 使用原始的预训练模型
        name = config
        model = get_encoder_by_name(name)
        return model


def get_encoder_by_name(name: str):
    if name == 'resnet50':
        model = ResNet50Encoder()
    elif name == 'resnet34':
        model = ResNet34Encoder()
    elif name == 'vgg16':
        model = Vgg16Encoder()
    elif name == 'mobilenetv3s':
        model = Mobile3SmallEncoder()
    elif name == 'efficientv2s':
        model = Efficient2SmallEncoder()
    elif name == 'efficientb2':
        model = Efficientb2Encoder()
    elif name == 'vitb16':
        model = ViTb16Encoder()
    elif name == 'vitb32':
        model = ViTb32Encoder()
    else:
        raise ValueError(f'no such model: {name}')
    return model

# 加载预训练的ResNet模型，删除最后的分类层
@singleton
class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        # 加载预训练的ResNet模型
        self.encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 删除最后的全连接层（分类层）
        self.encoder.fc = nn.Identity()
        # self.initialize()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x
            # x = x.half()
            # x = self.encoder(x)
            # return x.float()
            # return x

    def encode(self, x):
        return self.forward(x)


@singleton
class ResNet34Encoder(nn.Module):
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        # 加载预训练的ResNet模型
        self.encoder = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # 删除最后的全连接层（分类层）
        self.encoder.fc = nn.Identity()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x

    def encode(self, x):
        return self.forward(x)

@singleton
class Vgg16Encoder(nn.Module):
    def __init__(self):
        super(Vgg16Encoder, self).__init__()
        # 加载预训练的ResNet模型
        self.encoder = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # 删除最后的全连接层（分类层）
        self.encoder.fc = nn.Identity()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x

    def encode(self, x):
        return self.forward(x)

@singleton
class Mobile3SmallEncoder(nn.Module):
    def __init__(self):
        super(Mobile3SmallEncoder, self).__init__()
        # 加载预训练的ResNet模型
        self.encoder = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # 删除最后的全连接层（分类层）
        self.encoder.fc = nn.Identity()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x

    def encode(self, x):
        return self.forward(x)

@singleton
class Efficient2SmallEncoder(nn.Module):
    def __init__(self):
        super(Efficient2SmallEncoder, self).__init__()
        # 加载预训练的ResNet模型
        self.encoder = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        # 删除最后的全连接层（分类层）
        self.encoder.fc = nn.Identity()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x

    def encode(self, x):
        return self.forward(x)

@singleton
class Efficientb2Encoder(nn.Module):
    def __init__(self):
        super(Efficientb2Encoder, self).__init__()
        # 加载预训练的ResNet模型
        self.encoder = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        # 删除最后的全连接层（分类层）
        self.encoder.fc = nn.Identity()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x

    def encode(self, x):
        return self.forward(x)

@singleton
class ViTb16Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.vit_b_16(pretrained=True)
        self.initialize()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x

    def encoded(self, x):
        return self.forward(x)


@singleton
class ViTb32Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.vit_b_32(pretrained=True)
        self.initialize()

    def initialize(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x

    def encoded(self, x):
        return self.forward(x)



if __name__ == '__main__':
    # model = get_mae_encoder('resnet50', in_channels=1, ckpt_path=None)
    model = ViTb32Encoder()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.size())
    # model = get_encoder('resnet50')
