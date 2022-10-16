from typing import Any, Dict
import torch
import torchvision
from builder import ModelBuilder

from logger import BaseLogger, ConsoleLogger


class GANTrainer:
    def __init__(self, 
                 generator_builder: ModelBuilder,
                 discriminator_builder: ModelBuilder,
                 noise_distribution: torch.distributions.Distribution,
                 dataloader: torch.utils.data.DataLoader,
                 device: str,
                 logger: BaseLogger=ConsoleLogger(__name__)) -> None:
        self.device = device
        self.generator_builder = generator_builder
        self.discriminator_builder = discriminator_builder
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.logger = logger
        self.noise_distribution = noise_distribution
        
        self.true_label = self.make_true_label(self.batch_size)
        self.fake_label = self.make_fake_label(self.batch_size)
        self.criterion = torch.nn.BCELoss()
        self.generator, self.g_opt, self.g_scheduler = self.generator_builder.build()
        self.discriminator, self.d_opt, self.d_scheduler = self.discriminator_builder.build()
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.test_noise = self.make_noise(64, self.generator.input_dim)
        self.toImage = torchvision.transforms.ToPILImage()
        
    def make_noise(self, batch_size, latent_dim):
        return self.noise_distribution.sample((batch_size, latent_dim)).to(self.device)
    
    def make_true_label(self, batch_size):
        return torch.ones(batch_size, 1, device=self.device)
    
    def make_fake_label(self, batch_size):
        return torch.zeros(batch_size, 1, device=self.device)
    
    def train_discriminator(self, real_data, fake_data):
        self.d_opt.zero_grad()
        prediction_real = self.discriminator(real_data)
        error_real = self.criterion(prediction_real, self.true_label)
        error_real.backward()
        self.d_opt.step()
        
        self.d_opt.zero_grad()
        prediction_fake = self.discriminator(fake_data)
        error_fake = self.criterion(prediction_fake, self.fake_label)
        error_fake.backward()
        self.d_opt.step()
        
        for scheduler in self.d_scheduler:
            scheduler.step()
        
        return error_real + error_fake
    
    def train_generator(self, fake_data):
        self.g_opt.zero_grad()
        
        prediction = self.discriminator(fake_data)
        error = self.criterion(prediction, self.true_label)
        error.backward()
        
        self.g_opt.step()
        
        for scheduler in self.g_scheduler:
            scheduler.step()
        
        return error
    
    def _log(self, d_error, g_error, image_freq, current_epoch):
        # generate test image since GAN does not have performance guarantee
        imgs = self.generator(self.test_noise).cpu().detach()
        imgs = torchvision.utils.make_grid(imgs)
        msg = dict(d_loss=d_error,
                   g_loss=g_error,
                   generator=self.generator,
                   model_dir="model")
        if current_epoch % image_freq == 0:
            msg['image'] = self.toImage(imgs)
        self.logger.log(msg)
    
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
            
        return d_error / (i + 1), g_error / (i + 1)
    
    def run(self, 
            epochs: int=10, 
            num_train_dis: int=1):
        self.generator.train()
        self.discriminator.train()
        image_freq = int(epochs / 50) + 1 # ensure max 50 image log for wandb standard
        
        self.logger.on_epoch_start()
        for i in range(epochs):
            d_error, g_error = self.update_trainer(num_train_dis, self.dataloader)    
            self._log(d_error, g_error, image_freq, i)
        
        self.logger.on_epoch_end()