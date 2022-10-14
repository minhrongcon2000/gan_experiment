from sched import scheduler
from typing import Any, Dict, Type
import torch


class ModelBuilder:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: Type[torch.optim.Optimizer], 
                 optimizer_kwargs: Dict[str, Any],
                 scheduler: Type[torch.optim.lr_scheduler._LRScheduler]=None,
                 scheduler_kwargs: Dict[str, Any]=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        
    def build(self):
        if scheduler is None:
            return self.model, self.optimizer(self.model.parameters(), **self.optimizer_kwargs), None
        return self.model, self.optimizer(self.model.parameters(), **self.optimizer_kwargs), self.scheduler(self.optimizer, **self.scheduler_kwargs)