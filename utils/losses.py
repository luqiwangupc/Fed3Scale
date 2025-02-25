import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Any
import numpy as np
import seal
from seal import EncryptionParameters, SEALContext, KeyGenerator, CKKSEncoder, Encryptor, Decryptor, Evaluator


def get_loss_function(class_cri: str, consis_cri: str, num_class: int = None):
    class_cri = class_cri.lower()
    consis_cri = consis_cri.lower()
    classification_criterion = None
    consistency_criterion = None
    if class_cri == 'ce':
        classification_criterion = nn.CrossEntropyLoss()
    elif class_cri == 'encrypt':
        assert num_class is not None
        classification_criterion = EncryptLoss(num_class=num_class)
    else:
        raise ValueError(f"Unknown classification criterion: {class_cri}")
    if consis_cri == "mse":
        consistency_criterion = nn.MSELoss()
    elif consis_cri == "l1":
        consistency_criterion = nn.L1Loss()
    elif consis_cri == "KL":
        consistency_criterion = nn.KLDivLoss()
    else:
        raise ValueError(f"Unknown consistency criterion: {consis_cri}")
    return classification_criterion, consistency_criterion


class EncryptLoss(nn.Module):
    def __init__(self, num_class):
        super(EncryptLoss, self).__init__()
        self.num_class = num_class
        self.encrypt = Encrypt(num_classes=num_class)
        self.e_loss = ELoss

    def forward(self, outputs, labels):
        return self.e_loss(outputs, labels, self.encrypt)


class Encrypt:
    def __init__(self, num_classes):
        super(Encrypt, self).__init__()
        self.num_classes = num_classes

        parms = self.setup_ckks_parameters()

        # 创建SEAL上下文
        context = SEALContext(parms)
        # 生成公钥，私钥，加密器，解密器，评估器
        keygen = KeyGenerator(context)
        public_key = keygen.create_public_key()
        secrte_key = keygen.secret_key()

        self.encryptor = Encryptor(context, public_key)
        self.decryptor = Decryptor(context, secrte_key)
        self.evaluator = Evaluator(context)

        # 编码器用于将浮点数转换为明文
        self.encoder = CKKSEncoder(context)

        # 设置CKKS方案的缩放因子
        self.scale = 2 ** 40

    def setup_ckks_parameters(self):
        # 设置多项式模数和密钥大小
        parms = EncryptionParameters(seal.scheme_type.ckks)

        # 设置多项式环的度（多项式的系数数目必须是2的幂次）
        poly_modulus_degree = 8192
        parms.set_poly_modulus_degree(poly_modulus_degree)

        # 设置系数模数
        parms.set_coeff_modulus(seal.CoeffModulus.Create(poly_modulus_degree, [40, 40, 40, 40]))

        return parms

    def turn_to_ndarray(self, values):
        # 将其他类型（Torch.Tensor）转换为 np.ndarray
        if isinstance(values, torch.Tensor):
            if values.device != torch.device('cpu'):
                values = values.cpu()
            values = values.detach().numpy()
        if isinstance(values, list):
            values = np.array(values)
        values = values.flatten()
        return values

    def encrypt_ckks(self, values):
        # 编码为明文
        plain = self.encoder.encode(values, self.scale)

        # 加密明文，生成密文
        encrypted = self.encryptor.encrypt(plain)

        return encrypted

    def decrypt_ckks(self, encrypted):
        # 解密密文
        plain = self.decryptor.decrypt(encrypted)

        # 解码为浮点数
        decoded = self.encoder.decode(plain)

        return decoded

    def encrypt_labels(self, labels):
        # 将标签转为one-hot，再进行加密
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
        np_one_hot = self.turn_to_ndarray(one_hot)
        encrypted_labels = self.encrypt_ckks(np_one_hot)
        return encrypted_labels, one_hot


class ELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs: torch.Tensor, labels: torch.Tensor, encrypt: Encrypt):
        # 对标签进行加密
        encrypted_labels, onehot_labels = encrypt.encrypt_labels(labels)

        # 对模型输出进行加密
        np_outputs = encrypt.turn_to_ndarray(outputs)
        encrypted_outputs = encrypt.encrypt_ckks(np_outputs)

        # 在加密环境下计算损失 (output - labels)^2
        encrypted_loss = encrypt.evaluator.square(encrypt.evaluator.sub(encrypted_outputs, encrypted_labels))

        # 解密损失
        decrypted_loss = encrypt.decrypt_ckks(encrypted_loss)
        decrypted_loss = torch.from_numpy(decrypted_loss).to(outputs.device)
        ctx.save_for_backward(outputs, onehot_labels)
        return decrypted_loss.mean()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # 从forward中恢复保存的tensor
        outputs, labels = ctx.saved_tensors
        # 计算梯度 (outputs - labels)^2 的导数为 2 * (outputs - labels)
        grad_output = 2 * (outputs - labels) * grad_outputs[0]
        grad_input = -2 * (outputs - labels) * grad_outputs[0]
        grad_encrypt = None
        return grad_output, grad_input, grad_encrypt
