# Implement based on Ian Goodfellow repo. https://github.com/goodfeli/adversarial

import torch
import argparse
import torchvision
import numpy as np
from builder import ModelBuilder

from generator import MNISTGenerator
from discriminator import MNISTDiscriminator
from logger import ConsoleLogger, WandbLogger
from trainer import GANTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_train_discriminator", type=int, default=1)
parser.add_argument("--logger_type", type=str, default="console")
parser.add_argument("--wandb_api_key", type=str, required=False)
parser.add_argument("--project", type=str, default="vanilla_gan")
parser.add_argument("--run_name", type=str, default="GAN")
args = vars(parser.parse_args())

device = args["device"]

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

mnist = torchvision.datasets.MNIST("./data", download=True, transform=transforms)
dataloader = torch.utils.data.DataLoader(mnist, 
                                         batch_size=args["batch_size"], 
                                         shuffle=True)

logger = ConsoleLogger(__name__) if args['logger_type'] == 'console' else WandbLogger(__name__, args['wandb_api_key'], args['project'], args['run_name'])

generator = MNISTGenerator()
discriminator = MNISTDiscriminator()
noise_distribution = torch.distributions.Uniform(-np.sqrt(3.0), np.sqrt(3.0))
generator_builder = ModelBuilder(generator, torch.optim.SGD, dict(lr=0.1, momentum=0.5))
generator_builder.register_scheduler(torch.optim.lr_scheduler.ExponentialLR, dict(gamma=1 / 1.000004))
discriminator_builder = ModelBuilder(discriminator, torch.optim.SGD, dict(lr=0.1, momentum=0.5))
discriminator_builder.register_scheduler(torch.optim.lr_scheduler.ExponentialLR, dict(gamma=1 / 1.000004))
trainer = GANTrainer(generator_builder=generator_builder,
                     discriminator_builder=discriminator_builder,
                     device=device,
                     dataloader=dataloader,
                     logger=logger,
                     noise_distribution=noise_distribution)
trainer.run(epochs=args['epochs'], 
            num_train_dis=args['num_train_discriminator'])
