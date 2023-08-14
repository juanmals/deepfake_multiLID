import os
import torch
import numpy as np
import pdb
import argparse
from tqdm import tqdm
from diffusers import DDPMPipeline
from helper_extract.registrate_featuremaps import (
    extract_features, 
    extract_features_timm, 
    registrate_featuremaps
)

from helper_extract.utils import ( load_data )

from cfg import *

from misc import (
    args_handling,
    print_args,
    create_dir,
    save_to_pt,
    load_model,
    load_model_timm,
    normalize_images
)


TIMM = True

if __name__ == "__main__":
    #processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=['cifar10', 'mnist', 'butterfly32', 'oxfordflowers64', 'celebaHQ256'], help="")
    parser.add_argument("--extract", default="features", choices= ['normal', 'features',  'bb-fft', 'bb-multiLID', 'wb-multiLID'], help="")
    parser.add_argument("--mode",  default="nor", choices=['nor', 'adv'], help="normal, adv")
    # parser.add_argument("--model_ext", default="ema", help="ema")
    parser.add_argument("--nrsamples", default=2000, help="number samples")
    parser.add_argument("--model", default="rn18", help="ResNet18")

    parser.add_argument("--pretrained",    default="", help="pretrained")
    parser.add_argument("--load_gen_nor",  default="", help="load_gen_nor")
    parser.add_argument("--load_gen_adv",  default="", help="load_gen_adv")
    parser.add_argument("--save_extr_nor", default="", help="save_extr_nor")
    parser.add_argument("--save_extr_adv", default="", help="save_extr_adv")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file.')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()

    # args.load_json = os.path.join(cfg_extract_path, load_cfg)
    args = args_handling(args, parser, cfg_extract_path)
    print_args(args)

    if "timm" in args.save_extr_nor:
        TIMM = True
    print("TIMM> ", TIMM)

    print ("load model ...")
    if TIMM:
        model = load_model_timm(args)
    else:
        model = load_model(args)


    print("load data ...")
    nor_img, adv_img = load_data(args, ws_gen_path)


    if hasattr(args, "perturb_mode") and args.perturb_mode:
        print("perturb data ...")
        from helper_extract.perturbations import ( apply_transformation_to_datasets )
        
        nor_img, adv_img = apply_transformation_to_datasets(args, nor_img, adv_img)
    
    nor_img = normalize_images(args, nor_img)
    adv_img = normalize_images(args, adv_img)


    print("extract ... ")
    if args.extract == 'normal':
        # images are normalized now
        save_to_pt(args, ws_extract_path, nor_img, args.save_extr_nor)
        save_to_pt(args, ws_extract_path, adv_img, args.save_extr_adv)

        shape = nor_img.shape

    elif args.extract == 'features':
        nor_loader = torch.utils.data.DataLoader(nor_img, batch_size=args.nrsamples, shuffle=False, num_workers=4)
        adv_loader = torch.utils.data.DataLoader(adv_img, batch_size=args.nrsamples, shuffle=False, num_workers=4)

        if TIMM:
            activation_nor = extract_features_timm(args, nor_loader, model)
            nor = [args.layers, activation_nor]
            save_to_pt(args, ws_extract_path, nor, args.save_extr_nor)
            del nor, activation_nor

            activation_adv = extract_features_timm(args, adv_loader, model)
            adv = [args.layers, activation_adv]
            save_to_pt(args, ws_extract_path, adv, args.save_extr_adv)

            shape = 0

        else:
            get_layer_feature_maps_nor, layers_nor, model_nor, activation_nor = extract_features(args, nor_loader, model)
            nor = [layers_nor, activation_nor]
            save_to_pt(args, ws_extract_path, nor, args.save_extr_nor)

            get_layer_feature_maps_adv, layers_adv, model_adv, activation_adv = extract_features(args, adv_loader, model)
            adv = [layers_adv, activation_adv]
            save_to_pt(args, ws_extract_path, adv, args.save_extr_adv)

        shape = activation_adv[args.layers[-1]].shape

    elif args.extract == 'bb-fft':
        nor_loader = torch.utils.data.DataLoader(nor_img, batch_size=args.nrsamples, shuffle=False, num_workers=4)
        adv_loader = torch.utils.data.DataLoader(adv_img, batch_size=args.nrsamples, shuffle=False, num_workers=4)

        def apply_fft(im):
            im = im.float()
            im = im.cpu()
            im = im.data.numpy()
            fft = np.fft.fft2(im)
            fourier_spectrum = np.abs(fft)
            return fourier_spectrum

        nor_fft = []
        for img in nor_loader:
            fft_nor = apply_fft(img) # MFS
        save_to_pt(args, ws_extract_path, fft_nor, args.save_extr_nor)

        adv_fft = []
        for img in adv_loader:  
            fft_adv = apply_fft(img) # MFS
        save_to_pt(args, ws_extract_path, fft_adv, args.save_extr_adv)

        shape = adv_fft[-1].shape

    elif args.extract == 'wb-fft':
        pass

    elif args.extract in ['bb-multiLID', 'wb-multiLID']:
        
        from helper_extract.multiLID import multiLID_timm
        nor, adv = multiLID_timm(args, nor_img, adv_img, model)

        save_to_pt(args, ws_extract_path, nor, args.save_extr_nor)
        save_to_pt(args, ws_extract_path, adv, args.save_extr_adv)

    elif args.extract in ['LID']:
        from helper_extract.multiLID import multiLID_timm
        nor, adv = multiLID_timm(args, nor_img, adv_img, model)

        save_to_pt(args, ws_extract_path, nor, args.save_extr_nor)
        save_to_pt(args, ws_extract_path, adv, args.save_extr_adv)

    else:
        raise NotImplementedError


    print("nor.shape> ", nor.shape)
    print("adv.shape> ", adv.shape)
    print("extraction finished ...")