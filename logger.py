from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Union
import wandb
import torch
from datetime import datetime


MSG_OBJ_TYPE = Dict[str, Union[float, torch.Tensor]]


class LogType:
    INFO = "INFO"
    ERROR = "ERROR"


class BaseLogger(ABC):
    def __init__(self, logger_name) -> None:
        super().__init__()
        self.logger_name = logger_name
    
    @abstractmethod
    def on_epoch_start(self):
        pass
    
    @abstractmethod
    def log(self, msg_obj):
        pass
    
    @abstractmethod
    def on_epoch_end(self):
        pass
    

class ConsoleLogger(BaseLogger):
    def __init__(self, logger_name):
        super().__init__(logger_name)
        self.epoch_counter = 1
        
    def get_log(self, log_type, log_msg):
        return f'[{log_type}] - {self.logger_name} - {datetime.utcnow().isoformat()} - {log_msg}'
    
    def on_epoch_start(self):
        print(self.get_log(LogType.INFO, "Training Started!"))
    
    def log(self, msg_object):
        log_msg = 'Epoch {}, d_loss: {}, g_loss: {}'.format(self.epoch_counter,
                                                            msg_object['d_loss'],
                                                            msg_object['g_loss'])
        print(self.get_log(LogType.INFO, log_msg))
        self.epoch_counter += 1
        
    def on_epoch_end(self):
        print(self.get_log(LogType.INFO, "Training Finished!"))
        

class WandbLogger(BaseLogger):
    def __init__(self, logger_name):
        super().__init__(logger_name)

    def on_epoch_start(self, wandb_api_key, project_name, run_name):
        os.environ["WANDB_API_KEY"] = wandb_api_key
        if wandb.run is None:
            wandb.init(project=project_name, name=run_name)
        
    def log(self, msg_obj):
        msg_obj["image"] = wandb.Image(msg_obj["image"])
        wandb.log(msg_obj)
    
    def on_epoch_end(self):
        wandb.finish()
