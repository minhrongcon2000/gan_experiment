from sched import scheduler
from typing import Any, Dict, List, Type
import torch

from utils.scheduler import BaseScheduler


class ModelBuilder:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: Type[torch.optim.Optimizer], 
                 optimizer_kwargs: Dict[str, Any]) -> None:
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self.schedulers: List[BaseScheduler] = []
        
    def register_scheduler(self, 
                      scheduler: Type[BaseScheduler], 
                      scheduler_kwargs: Dict[str, Any]):
        new_scheduler = scheduler(self.optimizer, **scheduler_kwargs)
        self.schedulers.append(new_scheduler)
        
    def build(self):
        return self.model, self.optimizer, self.schedulers