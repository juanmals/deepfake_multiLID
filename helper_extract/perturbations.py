import argparse
import os
import torch

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
from io import BytesIO
from scipy.ndimage.filters import gaussian_filter

from tqdm import tqdm

# https://github.com/jonasricker/diffusion-model-deepfake-detection/blob/main/src/image_perturbations.py
# https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/create_perturbed_imagedata.py

def noise(image):
    # variance from U[5.0,20.0]
    variance = np.random.uniform(low=5., high=20.)
    image = np.copy(image).astype(np.float64)
    noise = variance * np.random.randn(*image.shape)
    image += noise
    return np.clip(image, 0., 255.).astype(np.uint8)


def blur(image, size=[3, 5, 7, 9]):
    # kernel_size from [1, 3, 5, 7, 9]
    kernel_size = size
    kernel_size = np.random.choice(kernel_size)
    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=cv2.BORDER_DEFAULT)


def jpeg(image, size=[]):
    # quality factor sampled from U[10, 75]
    if len(size) == 0:
        factor = np.random.randint(low=10, high=75)
        _, image = cv2.imencode(".jpg", image, [factor, 90])
    else:
        factor = size[0]
        _, image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), factor])

    return cv2.imdecode(image, 1)


def cropping(image):
    # crop between 5% and 20%
    percentage = np.random.uniform(low=.05, high=.2)
    x, y, _ = image.shape
    x_crop = int(x * percentage * .5)
    y_crop = int(y * percentage * .5)
    cropped = image[x_crop:-x_crop, y_crop:-y_crop]
    resized = cv2.resize(cropped, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    return resized


# https://github.com/PeterWang512/CNNDetection
def gaussian_blur(img, sigma):
    if type(sigma) == list:
        sigma = sigma[0]
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    return img

# https://github.com/PeterWang512/CNNDetection
def pil_jpg(img, compress_val):
    if type(compress_val) == list:
        compress_val = compress_val[0]
    out = BytesIO()
    #img = Image.fromarray(img)
    img = Image.fromarray((img*255).astype(np.uint8)).convert('RGB')

    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img).astype('float32')/255
    out.close()
    return img


def apply_to_dataset(args, image_functions, images):

    if torch.is_tensor(images):
        images = images.numpy()
    
    if images.shape[1] == 3:
        images = images.transpose((0,2,3,1))
    
    new_dataset = []

    for it in tqdm(range(images.shape[0]), total=images.shape[0]):
        cur_image = images[it]
        cur_function = image_functions.pop(0)

        if np.random.sample() > args.perturb_prob:
            if hasattr(args, "perturb_size"):
                new_image = cur_function(cur_image, args.perturb_size)
            else:
                new_image = cur_function(cur_image)
                assert not np.isclose(new_image, cur_image).all()
            cur_image = new_image.copy()
    
        new_dataset.append(cur_image)
        image_functions.append(cur_function)

    dataset = torch.from_numpy(np.stack(new_dataset).transpose((0,3,1,2)))

    return dataset


def apply_transformation_to_datasets(args, nor_img, adv_img):

    modes = args.perturb_mode

    image_functions = []
    for mode in modes:
        if mode == "noise":
            image_function = [noise]

        elif mode == "blur":
            image_function = [blur]

        elif mode == "jpeg":
            image_function = [jpeg]

        elif mode == "cropping":
            image_function = [cropping]

        elif mode == "combined":
            image_function = [noise, blur, jpeg, cropping]

        elif mode == "gaussian_blur":
            image_function = [gaussian_blur]

        elif mode == "pil_jpeg":
            image_function = [pil_jpg]
    
        else:
            raise NotImplementedError("Selected unrecognized mode: {mode}!")

        image_functions += image_function

    nor_img = apply_to_dataset(args, image_functions, nor_img)
    # breakpoint()
    # torch.save(nor_img,"/home/lorenzp/DeepfakeDetection/analysis/perturbations/nor_blurred.pt")
    adv_img = apply_to_dataset(args, image_functions, adv_img)
    # torch.save(nor_img,"/home/lorenzp/DeepfakeDetection/analysis/perturbations/adv_blurred.pt")
    
    return nor_img, adv_img