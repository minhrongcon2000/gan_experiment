import torch
import argparse
import torchvision
import numpy as np
from builder import ModelBuilder

from generator import DCGANGenerator
from discriminator import DCGANDiscriminator
from logger import ConsoleLogger, WandbLogger
from trainer import GANTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_train_discriminator", type=int, default=1)
parser.add_argument("--logger_type", type=str, default="console")
parser.add_argument("--wandb_api_key", type=str, required=False)
parser.add_argument("--project", type=str, default="dc_gan")
parser.add_argument("--run_name", type=str, default="GAN")
args = vars(parser.parse_args())

device = args["device"]

mean = 0.5
std = 0.5
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((mean, ), (std, )),
])

mnist = torchvision.datasets.MNIST("./data", 
                                 transform=transforms,
                                 download=True)

dataloader = torch.utils.data.DataLoader(mnist, 
                                         batch_size=args["batch_size"], 
                                         shuffle=True)

if args['logger_type'] == 'console':
    logger = ConsoleLogger(__name__) 
else: 
    logger = WandbLogger(__name__, 
                         args['wandb_api_key'], 
                         args['project'], 
                         args['run_name'])

generator = DCGANGenerator()
discriminator = DCGANDiscriminator()
noise_distribution = torch.distributions.Normal(loc=0., scale=1.)
generator_builder = ModelBuilder(generator, 
                                 torch.optim.Adam, 
                                 dict(lr=0.0002, betas=(0.5, 0.999)))
discriminator_builder = ModelBuilder(discriminator,
                                     torch.optim.Adam, 
                                     dict(lr=0.0002, betas=(0.5, 0.999)))
trainer = GANTrainer(generator_builder=generator_builder,
                     discriminator_builder=discriminator_builder,
                     device=device,
                     dataloader=dataloader,
                     logger=logger,
                     noise_distribution=noise_distribution)
trainer.run(epochs=args['epochs'], 
            num_train_dis=args['num_train_discriminator'], 
            post_process=lambda img: 0.5 * img + 0.5)
