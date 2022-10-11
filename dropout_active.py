import glob
import cv2
import shutil
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
import albumentations as albu
import sys
from skimage.io import imread

Image.MAX_IMAGE_PIXELS = 93312000000

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model_path",
                    help="path to model", metavar="FILE")
parser.add_argument("-i", "--input", dest="input",
                    help="path to input folder")
parser.add_argument("-b", "--backbone", dest="backbone",
                    help="backbone name")

args = parser.parse_args()
print("arguments are ", args)

filenames = glob.glob(args.input + "/*.jpg") + glob.glob(args.input + "/*.png")

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

preprocess_fn = get_preprocessing_fn(args.backbone, pretrained='imagenet')
test_augmentation = albu.Compose([
    albu.augmentations.transforms.PixelDropout(always_apply=True)
    # albu.PadIfNeeded(min_height=INPUT_SIZE, min_width=INPUT_SIZE, always_apply=True, border_mode=0),
    # albu.Resize(height=INPUT_SIZE, width=INPUT_SIZE)
])

model = torch.load(args.model_path)
model.eval()
preprocess_transform = albu.Compose([
    albu.Lambda(image=preprocess_fn),
    albu.Lambda(image=to_tensor, mask=to_tensor)
])

preprocess_im = albu.Compose([
    albu.Lambda(image=preprocess_fn),
    albu.Lambda(image=to_tensor)
])

m = torch.nn.Sigmoid()

for filename in filenames:
    filename = filename.replace("\\", "/")
    split = filename.split("/")
    im_name_original = split[-1]
    im_name = im_name_original[:im_name_original.find(".")]

    im = imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

    im = im.astype('uint8')

    inf = test_augmentation(image=im)['image']
    inf = preprocess_fn(inf)

    print(inf.shape)

    inf = to_tensor(inf)
    inf = np.expand_dims(inf, 0)

    print(inf.shape)
    # sys.exit()

    with torch.no_grad():
        inf = torch.from_numpy(inf).cuda().float()
        mask = model(inf)
        # mask = m(mask)

    print(mask)
