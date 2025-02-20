
import torch.nn as nn


def get_loss_function(class_cri: str, consis_cri: str):
    class_cri = class_cri.lower()
    consis_cri = consis_cri.lower()
    if class_cri == 'ce':
        classification_criterion = nn.CrossEntropyLoss()
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
