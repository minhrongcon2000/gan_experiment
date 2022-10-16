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
        

class WandbLogger(ConsoleLogger):
    def __init__(self, logger_name, wandb_api_key, project_name=None, run_name=None):
        super().__init__(logger_name)
        self.project_name = project_name
        self.run_name = run_name
        self.wandb_api_key = wandb_api_key

    def on_epoch_start(self):
        super().on_epoch_start()
        os.environ["WANDB_API_KEY"] = self.wandb_api_key
        if wandb.run is None:
            wandb.init(project=self.project_name, name=self.run_name)
        
    def log(self, msg_obj):
        super().log(msg_obj)
        if "image" in msg_obj:
            msg_obj["image"] = wandb.Image(msg_obj["image"])
        if not os.path.exists(msg_obj["model_dir"]):
            os.makedirs(msg_obj["model_dir"])
        torch.save(msg_obj["generator"].state_dict(), os.path.join(msg_obj["model_dir"], "generator.pth"))
        wandb.log(dict(
            d_loss=msg_obj['d_loss'],
            g_loss=msg_obj['g_loss'],
            image=msg_obj["image"]
        ))
        wandb.save(os.path.join(msg_obj["model_dir"], "generator.pth"))
    
    def on_epoch_end(self):
        super().on_epoch_end()
        wandb.finish()
