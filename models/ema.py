from copy import deepcopy
import torch


class EMA(torch.nn.Module):
    def __init__(self, model, decay, dynamic_decay=False):
        super().__init__()
        self.model = deepcopy(model)
        self.decay = decay
        self.initialize()
        self.dynamic_decay = dynamic_decay
        self.update_counts = 0
        # self.update_steps = update_steps

    def initialize(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def update_by_model(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * param.data
        self.update_decay()

    def update_by_parameters(self, parameters):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), parameters):
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * param
        self.update_decay()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update_decay(self):
        if not self.dynamic_decay:
            return
        # cifar10
        if self.update_counts == 0:
            update_steps = 0.1
        elif self.update_counts == 1:
            update_steps = 0.1
        elif self.update_counts == 2:
            update_steps = 0.2
        else:
            update_steps = 0.4
        update_steps = max(0.01, update_steps)
        new_decay = self.decay + update_steps * self.update_counts
        self.update_counts += 1
        self.decay = min(0.99, new_decay)


if __name__ == '__main__':
    update_counts=0
    update_steps = 0.1
    decay = 0.1
    for update_counts in range(0, 9):
        if update_counts == 0:
            update_steps = 0.1
        elif update_counts == 1:
            update_steps = 0.1
        elif update_counts == 2:
            update_steps = 0.2
        else:
            update_steps = 0.4
        update_steps = max(0.01, update_steps)
        new_decay = decay + update_steps * update_counts
        # update_counts += 1
        decay = min(0.99, new_decay)
        print(decay)