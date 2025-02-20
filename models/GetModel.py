from models.encoder import get_encoder
from models.classifier import get_classifier
from models.ema import EMA


def get_model(level, config):
    """
    根据Tree的等级（云、边、端）获取模型（EMA、Classifier、Encoder）
    :param config: 模型的配置（默认为OmegaConf）
    :param level: 当前的Level
    :return: Model
    """
    if level == 0:  # Cloud
        classifier = get_classifier(name=config.models.classifier_name)
        ema_classifier = EMA(classifier, decay=config.models.ema_decay, dynamic_decay=config.models.emd_dynamic_decay)
        del classifier
        ema_classifier.eval()
        return ema_classifier
    elif level == 1:    # Edge
        classifier = get_classifier(name=config.models.classifier_name)
        classifier.train()
        return classifier
    elif level == 2:    # End
        encoder = get_encoder(config)
        encoder.eval()
        return encoder
    else:
        raise ValueError(f"level {level} is not supported")

