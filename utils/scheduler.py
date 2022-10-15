from abc import ABC, abstractmethod
import torch


class BaseScheduler(ABC):
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        super().__init__()
        self.optimizer = optimizer
    
    @abstractmethod
    def step(self):
        pass


class MomentumScheduler(BaseScheduler):
    def __init__(self, 
                 optimizer: torch.optim.Optimizer, 
                 saturate: float, 
                 final_momentum: float, 
                 start: int) -> None:
        super().__init__(optimizer)
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
            

class ExponentialLRScheduler(BaseScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, decay_factor: float, min_lr: float) -> None:
        super().__init__(optimizer)
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        
    def step(self):
        for group in self.optimizer.param_groups:
            current_lr = group['lr']
            new_lr = max(current_lr * self.decay_factor, self.min_lr)
            group['lr'] = new_lr
