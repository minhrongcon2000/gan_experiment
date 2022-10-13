import torch
import argparse
import torchvision

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
    torchvision.transforms.Normalize((0,), (255,))
])

mnist = torchvision.datasets.MNIST("./data", download=True, transform=transforms)
dataloader = torch.utils.data.DataLoader(mnist, 
                                         batch_size=args["batch_size"], 
                                         shuffle=True)

logger = ConsoleLogger(__name__) if args['logger_type'] == 'console' else WandbLogger(__name__, args['wandb_api_key'], args['project'], args['run_name'])

generator = MNISTGenerator()
discriminator = MNISTDiscriminator()
trainer = GANTrainer(generator=generator,
                     discriminator=discriminator,
                     device=device,
                     dataloader=dataloader,
                     logger=logger,
                     d_optimizer=torch.optim.SGD,
                     g_optimizer=torch.optim.SGD,
                     g_opt_kwargs=dict(lr=0.1, momentum=0.5),
                     d_opt_kwargs=dict(lr=0.1, momentum=0.5))
trainer.run(epochs=args['epochs'], 
            num_train_dis=args['num_train_discriminator'])
