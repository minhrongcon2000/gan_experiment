import math
import torch
import argparse
import torchvision
from builder import ModelBuilder

from generator import DCGANGenerator
from discriminator import DCGANDiscriminator
from logger import ConsoleLogger, WandbLogger
from trainer import WGANTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_train_discriminator", type=int, default=5)
parser.add_argument("--logger_type", type=str, default="console")
parser.add_argument("--wandb_api_key", type=str, required=False)
parser.add_argument("--project", type=str, default="dc_wgan")
parser.add_argument("--run_name", type=str, default="GAN")
parser.add_argument("--weight_clip", type=float, default=0.01)
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


total_batch_per_epoch = math.ceil(len(mnist) / args["batch_size"])
image_freq = math.ceil(args["epochs"] * total_batch_per_epoch / 50)

if args['logger_type'] == 'console':
    logger = ConsoleLogger(__name__) 
else: 
    logger = WandbLogger(__name__, 
                         args['wandb_api_key'], 
                         args['project'], 
                         args['run_name'])

generator = DCGANGenerator()
discriminator = DCGANDiscriminator(sigmoid_applied=False)
noise_distribution = torch.distributions.Normal(loc=0., scale=1.)
generator_builder = ModelBuilder(generator, 
                                 torch.optim.RMSprop, 
                                 dict(lr=0.00005))
discriminator_builder = ModelBuilder(discriminator,
                                     torch.optim.Adam, 
                                     dict(lr=0.00005))
trainer = WGANTrainer(generator_builder=generator_builder,
                     discriminator_builder=discriminator_builder,
                     device=device,
                     dataloader=dataloader,
                     logger=logger,
                     noise_distribution=noise_distribution,
                     clip=args["weight_clip"])
trainer.run(epochs=args['epochs'], 
            num_train_dis=args['num_train_discriminator'], 
            post_process=lambda img: 0.5 * img + 0.5,
            image_freq=image_freq)
