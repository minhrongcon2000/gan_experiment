import torch
import argparse
import torchvision

from generator import MNISTGenerator
from discriminator import MNISTDiscriminator
from trainer import GANTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--num_train_discriminator", type=int, default=1)
args = vars(parser.parse_args())

device = args["device"]

transforms = torchvision.transforms.Compose(
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (255,))
)

mnist = torchvision.datasets.MNIST("./data", download=True)
dataloader = torch.utils.data.DataLoader(mnist, 
                                         batch_size=args["batch_size"], 
                                         shuffle=True,
                                         transforms=transforms)

generator = MNISTGenerator()
discriminator = MNISTDiscriminator()
trainer = GANTrainer(generator=generator,
                     discriminator=discriminator,
                     device=device)
trainer.run(epochs=args['epochs'], 
            num_train_dis=args['num_train_discriminator'], 
            dataloader=dataloader)
