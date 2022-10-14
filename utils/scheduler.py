import torch


class MomentumScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, saturate: float, final_momentum: float, start: int) -> None:
        self.optimizer = optimizer
        self.saturate = saturate
        self.final_momentum = final_momentum
        self.start = start
        self._step_counter = 0
        
    def step(self):
        self._step_counter += 1
        alpha = (self._step_counter - self.start) / (self.saturate - self.start)
        alpha = torch.clamp(alpha, 0, 1)
        for group in self.optimizer.param_groups:
            group['momentum'] = (1 - alpha) * group['momentum'] + alpha * self.final_momentum