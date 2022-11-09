from typing import Any, Dict
import torch
import torchvision
from builder import ModelBuilder

from logger import BaseLogger, ConsoleLogger
from utils.loss import WGANLoss


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
        true_label = self.make_true_label(real_data.size(0))
        fake_label = self.make_fake_label(fake_data.size(0))
        
        prediction_real = self.discriminator(real_data)
        self.d_opt.zero_grad()
        error_real = self.criterion(prediction_real, true_label)
        error_real.backward()
        self.d_opt.step()
        
        prediction_fake = self.discriminator(fake_data)
        self.d_opt.zero_grad()
        error_fake = self.criterion(prediction_fake, fake_label)
        error_fake.backward()
        self.d_opt.step()
        
        for scheduler in self.d_scheduler:
            scheduler.step()
        
        return error_real.item() + error_fake.item()
    
    def train_generator(self, fake_data):
        true_label = self.make_true_label(fake_data.size(0))
        prediction = self.discriminator(fake_data)
        
        self.g_opt.zero_grad()
        error = self.criterion(prediction, true_label)
        error.backward()
        self.g_opt.step()
        
        for scheduler in self.g_scheduler:
            scheduler.step()
        
        return error.item()
    
    def _log(self, 
             epoch,
             d_error, 
             g_error, 
             image_freq, 
             current_timestep,
             post_process=None):
        # generate test image since GAN does not have performance guarantee
        imgs = self.generator(self.test_noise).cpu().detach()
        if post_process is not None:
            imgs = post_process(imgs)
        imgs = torchvision.utils.make_grid(imgs)
        msg = dict(epoch=epoch,
                   d_loss=d_error,
                   g_loss=g_error,
                   generator=self.generator,
                   current_timestep=current_timestep,
                   model_dir="model")
        if current_timestep % image_freq == 0:
            msg['image'] = self.toImage(imgs)
        self.logger.log(msg)
    
    def update_trainer(self, 
                       num_train_dis, 
                       dataloader, 
                       post_process, 
                       image_freq,
                       epoch):
        g_error = 0
        d_error = 0
        
        for i, (imgs, _) in enumerate(dataloader):
            for _ in range(num_train_dis):
                # Train discriminator first with some degree of update
                fake_data = self.generator(self.make_noise(imgs.size(0), 
                                                        self.generator.input_dim)).detach()
                real_data = imgs.to(self.device)
                d_error = self.train_discriminator(real_data, fake_data)
                
            # Train generator afterwards
            fake_data = self.generator(self.make_noise(imgs.size(0), self.generator.input_dim))
            g_error = self.train_generator(fake_data)
            self._log(epoch,
                      d_error, 
                      g_error, 
                      image_freq, 
                      current_timestep=i,
                      post_process=post_process)
    
    def run(self, 
            epochs: int=10, 
            num_train_dis: int=1,
            image_freq: int=1,
            post_process=None):
        self.generator.train()
        self.discriminator.train()
        
        self.logger.on_epoch_start()
        for epoch in range(epochs):
            self.update_trainer(num_train_dis, 
                                self.dataloader, 
                                post_process, 
                                image_freq,
                                epoch)
        self.logger.on_epoch_end()
        
        
class WGANTrainer(GANTrainer):
    def __init__(self, 
                 generator_builder: ModelBuilder, 
                 discriminator_builder: ModelBuilder, 
                 noise_distribution: torch.distributions.Distribution, 
                 dataloader: torch.utils.data.DataLoader, 
                 device: str, 
                 logger: BaseLogger = ConsoleLogger(__name__),
                 clip: float=0.01) -> None:
        super().__init__(generator_builder, 
                         discriminator_builder, 
                         noise_distribution, 
                         dataloader, 
                         device, 
                         logger)
        self.clip = clip
        self.criterion = WGANLoss()
        
    def train_generator(self, fake_data):
        prediction_fake = self.discriminator(fake_data)
        
        self.g_opt.zero_grad()
        error = self.criterion(prediction_fake)
        error.backward()
        self.g_opt.step()
        
        for scheduler in self.g_scheduler:
            scheduler.step()
        
        return error.item()
    
    def train_discriminator(self, real_data, fake_data):
        prediction_real = self.discriminator(real_data)
        prediction_fake = self.discriminator(fake_data)
        
        self.d_opt.zero_grad()
        error_d = -self.criterion(prediction_fake, prediction_real)
        error_d.backward()
        self.d_opt.step()
        
        for scheduler in self.d_scheduler:
            scheduler.step()
            
        # clip parameter of discriminator
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.clip, self.clip)
        
        return -error_d.item()
    

class WGANGPTrainer(WGANTrainer):
    def __init__(self, 
                 generator_builder: ModelBuilder, 
                 discriminator_builder: ModelBuilder, 
                 noise_distribution: torch.distributions.Distribution, 
                 dataloader: torch.utils.data.DataLoader, 
                 device: str, 
                 logger: BaseLogger = ConsoleLogger(__name__), 
                 clip: float = 0.01,
                 l: float=10.) -> None:
        super().__init__(generator_builder, 
                         discriminator_builder, 
                         noise_distribution, 
                         dataloader, 
                         device, 
                         logger, 
                         clip)
        self.l = l
        
    def train_discriminator(self, real_data, fake_data):
        prediction_real = self.discriminator(real_data)
        prediction_fake = self.discriminator(fake_data)
        N, C, H, W = real_data.shape
        eps = torch.rand(N, 1, 1, 1).repeat(1, C, H, W).to(self.device)
        interpolation = eps * prediction_real + (1 - eps) * prediction_fake
        prediction_interpolate = self.discriminator(interpolation)
        
        self.d_opt.zero_grad()
        error_d = -self.criterion(prediction_fake, 
                                  prediction_real, 
                                  interpolate=(interpolation, 
                                               prediction_interpolate),
                                  l=self.l)
        error_d.backward()
        self.d_opt.step()
        
        for scheduler in self.d_scheduler:
            scheduler.step()
        
        return error_d.item()