from typing import Type
import torch
import torchvision

from logger import BaseLogger, ConsoleLogger


class GANTrainer:
    def __init__(self, 
                 generator: torch.nn.Module, 
                 discriminator: torch.nn.Module, 
                 device: str, 
                 dataloader: torch.utils.data.DataLoader,
                 g_lr: float=2e-4, 
                 d_lr: float=2e-4,
                 g_optimizer: Type[torch.optim.Optimizer]=torch.optim.Adam,
                 d_optimizer: Type[torch.optim.Optimizer]=torch.optim.Adam,
                 logger: BaseLogger=ConsoleLogger(__name__)):
        self.device = device
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.logger = logger
        
        self.true_label = self.make_true_label(self.batch_size)
        self.fake_label = self.make_fake_label(self.batch_size)
        self.criterion = torch.nn.BCELoss()
        self.g_opt = g_optimizer(self.generator.parameters(), lr=self.g_lr)
        self.d_opt = d_optimizer(self.discriminator.parameters(), lr=self.d_lr)
        self.test_noise = self.make_noise(64, self.generator.input_dim)
        self.toImage = torchvision.transforms.ToPILImage()
        
    def make_noise(self, batch_size, latent_dim):
        return torch.randn(batch_size, latent_dim, device=self.device)
    
    def make_true_label(self, batch_size):
        return torch.ones(batch_size, 1, device=self.device)
    
    def make_fake_label(self, batch_size):
        return torch.zeros(batch_size, 1, device=self.device)
    
    def train_discriminator(self, real_data, fake_data):
        self.d_opt.zero_grad()
        
        prediction_real = self.discriminator(real_data)
        error_real = self.criterion(prediction_real, self.true_label)
        error_real.backward()
        
        prediction_fake = self.discriminator(fake_data)
        error_fake = self.criterion(prediction_fake, self.fake_label)
        error_fake.backward()
        
        self.d_opt.step()
        
        return error_real + error_fake
    
    def train_generator(self, fake_data):
        self.g_opt.zero_grad()
        
        prediction = self.discriminator(fake_data)
        error = self.criterion(prediction, self.true_label)
        error.backward()
        
        self.g_opt.step()
        
        return error
    
    def update_trainer(self, num_train_dis, dataloader):
        g_error = 0
        d_error = 0
        
        for i, (imgs, _) in enumerate(dataloader):
            # Train discriminator first with some degree of update
            for _ in range(num_train_dis):
                fake_data = self.generator(self.make_noise(self.batch_size, 
                                                           self.generator.input_dim)).detach()
                real_data = imgs.to(self.device)
                d_error += self.train_discriminator(real_data, fake_data) / num_train_dis
                
            # Train generator afterwards
            fake_data = self.generator(self.make_noise(self.batch_size, self.generator.input_dim))
            g_error += self.train_generator(fake_data)
            
        # generate test image since GAN does not have performance guarantee
        imgs = self.generator(self.test_noise).cpu().detach()
        imgs = torchvision.utils.make_grid(imgs)
        self.logger.log(dict(d_loss=d_error / (i + 1),
                        g_loss=g_error / (i + 1),
                        image=self.toImage(imgs)))
    
    def run(self, 
            epochs: int=10, 
            num_train_dis: int=1):
        self.generator.train()
        self.discriminator.train()
        
        self.logger.on_epoch_start()
        for _ in range(epochs):
            self.update_trainer(num_train_dis, self.dataloader)    
        
        self.logger.on_epoch_end()