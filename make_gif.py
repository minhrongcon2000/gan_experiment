import wandb
import argparse
from PIL import Image
import os


def images_to_gif(image_fnames, fname):
    image_fnames.sort(key=lambda x: int(x.split('_')[-2])) #sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
               save_all=True, loop=0)



parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str)
parser.add_argument("--wandb_api_key", type=str)
args = parser.parse_args()

os.environ["WANDB_API_KEY"] = args.wandb_api_key

api = wandb.Api()
run = api.run(args.run_id)

if not os.path.exists("media/images"):
    for file in run.files():
        if file.name.endswith('.png'):
            file.download()

image_filenames = [os.path.join("media/images", item) for item in os.listdir("media/images")]
images_to_gif(image_filenames, "dcgan")
