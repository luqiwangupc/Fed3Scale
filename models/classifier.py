import torch
import torch.nn as nn
from einops import repeat


def get_classifier(name: str):
    if name == 'small-transformer':
        return SmallClassifier(input_dim=2048, num_classes=10)
    elif name == 'medium-transformer':
        return MediumClassifier(input_dim=2048, num_classes=10)
    elif name == 'large-transformer':
        return LargeClassifier(input_dim=2048, num_classes=10)
    elif name == 'small-mlp':
        return SmallTransformerMLPClassifier(input_dim=2048, num_classes=10)
    elif name == 'medium-mlp':
        return MediumTransformerMLPClassifier(input_dim=2048, num_classes=100)
    elif name == 'large-mlp':
        return LargeTransformerMLPClassifier(input_dim=2048, num_classes=10)
    else:
        raise ValueError(f'no such classifier {name}')


# Transformer Encoder 模块 (用于将 [batch_size, 1024] 输入转换)
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1024, num_layers=6, num_heads=8, hidden_dim=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x 的形状为 [batch_size, input_dim]
        # 我们在这里添加一个假的时间维度，将其视为 [1, batch_size, input_dim] 以符合 Transformer 的输入要求
        x = x.unsqueeze(0)  # [batch_size, input_dim] -> [1, batch_size, input_dim]
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # 变回 [batch_size, input_dim]
        return x


# Large 分类器
class LargeClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=10, dropout=0.2):
        super(LargeClassifier, self).__init__()
        self.transformer = TransformerEncoder(input_dim=input_dim, num_layers=6, num_heads=8, hidden_dim=2048, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, int(input_dim / 2))  # 分类层
        self.fc2 = nn.Linear(int(input_dim / 2), num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Medium 分类器
class MediumClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=10):
        super(MediumClassifier, self).__init__()
        self.transformer = TransformerEncoder(input_dim=input_dim, num_layers=4, num_heads=4, hidden_dim=1024)  # 剪枝
        self.fc = nn.Linear(input_dim, num_classes)  # 分类层

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)


# Small 分类器
class SmallClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=10, dropout=0.2):
        super(SmallClassifier, self).__init__()
        self.transformer = TransformerEncoder(input_dim=input_dim, num_layers=2, num_heads=2, hidden_dim=512)  # 进一步剪枝
        self.fc = nn.Linear(input_dim, num_classes)  # 分类层

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)


class ViT_Latent(nn.Module):
    def __init__(self, latent_dim=768, num_classes=1000, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super(ViT_Latent, self).__init__()

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, latent_dim + 1, latent_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=mlp_dim,
                                                dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, D = x.shape  # Latent input [B, D]

        # Expand dimension to match transformer input format [B, num_patches, embed_dim]
        x = x.unsqueeze(1)  # [B, 1, D]

        # Concatenate class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + latent_dim, D]

        # Add position embedding
        x = x + self.pos_embedding[:, :x.shape[1], :]  # [B, 1 + latent_dim, D]

        # Apply transformer
        x = self.transformer(x)  # [B, num_patches + 1, D]

        # Class token output
        x = x[:, 0]  # [B, D]

        # Classification head
        x = self.mlp_head(x)  # [B, num_classes]

        return x


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=576, hidden_dim1=256, hidden_dim2=128, num_classes=10, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()

        # 第一层：全连接层 + Batch Normalization + ReLU 激活 + Dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # 第二层：全连接层 + Batch Normalization + ReLU 激活 + Dropout
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # 输出层：全连接层（分类层）
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        # 第一层前向传播
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 第二层前向传播
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # 输出层
        x = self.fc3(x)
        return x


class SmallTransformerMLPClassifier(nn.Module):
    def __init__(self, input_dim=576, hidden_dim1=256, hidden_dim2=128, num_classes=10, dropout_rate=0.5):
        super(SmallTransformerMLPClassifier, self).__init__()
        self.transformer_encoder = TransformerEncoder(input_dim, num_layers=1, num_heads=1, hidden_dim=1024, dropout=dropout_rate)
        self.classfier = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.classfier(x)
        return x


class MediumTransformerMLPClassifier(nn.Module):
    def __init__(self, input_dim=576, hidden_dim1=1024, hidden_dim2=512, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(input_dim, num_layers=4, num_heads=4, hidden_dim=2048, dropout=dropout_rate)
        self.classfier = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.classfier(x)
        return x

class LargeTransformerMLPClassifier(nn.Module):
    def __init__(self, input_dim=576, hidden_dim1=2048, hidden_dim2=1024, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(input_dim, num_layers=6, num_heads=8, hidden_dim=2048, dropout=dropout_rate)
        self.classfier = MLPClassifier(input_dim, hidden_dim1, hidden_dim2, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.classfier(x)
        return x


if __name__ == '__main__':
    # 示例
    inputs = torch.randn(64, 1024)  # [batch_size, 1024]

    # # 实例化分类器
    # small_classifier = SmallClassifier(num_classes=10)
    # medium_classifier = MediumClassifier(num_classes=10)
    # large_classifier = LargeClassifier(num_classes=10)
    #
    # # 获取分类结果
    # small_output = small_classifier(inputs)  # Small 分类器
    # medium_output = medium_classifier(inputs)  # Medium 分类器
    # large_output = large_classifier(inputs)  # Large 分类器
    #
    # print(small_output.shape)  # 输出 [64, 10]
    # print(medium_output.shape)  # 输出 [64, 10]
    # print(large_output.shape)  # 输出 [64, 10]

    vit_classifier = ViT_Latent(latent_dim=1024, num_classes=10, num_heads=8)

    vit_output = vit_classifier(inputs)
    print(vit_output.shape)
