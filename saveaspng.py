import os, pdb
import numpy as np
import matplotlib.image as mpimg
import torch
from os.path import exists

from cfg import * 


def save_to_png(original_data, pth):
    for i, data in enumerate(original_data):
        mpimg.imsave(os.path.join(pth, str(i) + '.png'), data)


def torch2hwcuint8(x, clip=False, flip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = x.detach().cpu()

    if flip:
        breakpoint()
        x = torch.flip(x, [3,2])

    x = x.permute(0,2,3,1)
    print("shape> ", x.shape)
    # x = (x+1.0)*127.5
    x = x * 255
    x = x.numpy().astype(np.uint8)
    return x


def save(x, format_string, start_idx=0, clip=False, flip=False):
    import os, PIL.Image
    os.makedirs(os.path.split(format_string)[0], exist_ok=True)
    x = torch2hwcuint8(x, clip=clip, flip=flip)

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    for i in range(x.shape[0]):
        PIL.Image.fromarray(x[i]).save(format_string.format(start_idx+i))


if __name__ == "__main__":
    print("start!")

    # pt_files = [
    #     "cifar10/nor_train.pt",
    #     "cifar10/adv_ddpm.pt",
    #     "cifar10/adv_ddim.pt",
    #     "cifar10/adv_pndm.pt",
    # ]

    # pt_files = [
    #     "mnist/nor_train.pt",
    #     "mnist/adv_ddpm.pt",
    # ]

    # pt_files = [
    #     # "mnist/nor_train.pt",
    #     # "mnist/adv_ddpm.pt",
    #     # "mnist/adv_ddpm2.pt",
    # ]

    # pt_files = [
    #     "oxfordflowers64/nor_train.pt",
    #     "oxfordflowers64/adv_ddpm.pt",
    # ]

    # pt_files = [
    #     # "butterflies128/nor_train.pt",
    #     # "butterflies128/adv_ddpm_ema.pt",
    # ]


    # pt_files = [
    #     "celebaHQ128/nor_train_haircolor.pt",
    #     "celebaHQ128/adv_ddpm.pt",
    # ]

    pt_files = [
       #"celebaHQ256/nor_train_haircolor.pt",
       #"celebaHQ256/adv_ddpm_ema.pt",
       #"celebaHQ256/adv_ddim_ema.pt",
       "celebaHQ256/adv_ldm.pt",
    #    "lsun-bedroom256/nor_train.pt",
       #"lsun-bedroom256/adv_ddpm_ema.pt",
       #"lsun-bedroom256/adv_ddim_ema.pt",
       #"lsun-bedroom256/adv_pndm_ema.pt",
    #    "lsun-cat256/nor_train.pt",
       #"lsun-cat256/adv_ddpm_ema.pt",
       #"lsun-cat256/adv_ddim_ema.pt",
    #    "lsun-cat256/adv_pndm_ema.pt",
    #    "lsun-church256/nor_train.pt",
       #"lsun-church256/adv_ddpm_ema.pt",
       #"lsun-church256/adv_ddim_ema.pt",
       #"lsun-church256/adv_pndm_ema.pt"
    ]





    base_path = "/home/lorenzp/workspace/DeepfakeDetection/results/gen"

    for tmp_file in pt_files:
        # tmp = pt_files[0]
        tmp = tmp_file
        tmp_png =  tmp.replace(".pt", "_png")

        print("from .pt to .png: ", tmp)

        pt_file = os.path.join(base_path, tmp)

        if not exists(pt_file):
            print("Err: Not found: ", tmp)
            continue

        output_folder = os.path.join(base_path, tmp_png)
        out_exists = os.path.isdir(output_folder)
        if out_exists:
            print("Info> Check the output folder: ", output_folder)
            breakpoint()

        x = torch.load(pt_file)
        format_string = output_folder + "/{:06}.png"
        save(x, format_string, start_idx=0, flip=False)

    print("finished!")
