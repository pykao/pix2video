from tqdm.notebook import tqdm
import torch
from mydiffusers import StableDiffusionPipeline,StableDiffusionDepth2ImgPipeline,DDIMScheduler
import numpy as np
import abc
import shutil
from PIL import Image
from pytorch_lightning import seed_everything
import os
import cv2
import contextlib
import imageio
import argparse
import configparser

def create_video_with_images(glob_pattern: str, output_filename: str):
    """
    Create a video from a set of images using ffmpeg.
    Images are gathered using a glob pattern (e.g. "img_????.png").
    """
    os.system(f"ffmpeg -pattern_type glob -i '{glob_pattern}' -c:v libx264 -r 30 -pix_fmt yuv420p {output_filename}")

def create_gif_with_images(glob_pattern: str, output_filename: str):
    """
    Create a video from a set of images using ffmpeg.
    Images are gathered using a glob pattern (e.g. "img_????.png").
    """
    os.system(f"ffmpeg -f image2 -framerate 30 -pattern_type glob -i '{glob_pattern}' -loop -1 {output_filename}")

########Stable Diffusion parameters##########

parser = argparse.ArgumentParser()

#######This file is to test the setup where we do not do any finetuning and we use the cfg update############
parser.add_argument("--input_path", type=str, default='../example_data/blackswan',help="input path")

opt = parser.parse_args()

input_path = opt.input_path
print("INPUT_PATH: ", input_path)

glob_pattern = os.path.join(input_path, './*.png')
output = os.path.join(input_path, 'out_cfg.gif')
create_gif_with_images(glob_pattern, output)
output = os.path.join(input_path, 'out_cfg.mp4')
create_video_with_images(glob_pattern, output)
