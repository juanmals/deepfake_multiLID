import os
import torch
import numpy as np
import natsort 

from cfg import *


def change_dict_device(dictionary, device='cpu'):

    todevice = torch.device(device)

    for the_key, the_value in dictionary.items():
        # print ( the_key, 'corresponds to', the_value.cpu())
        dictionary[the_key] = the_value.to(todevice)

    return dictionary


def feature_dict_to_list_timm(args, feature_dict):

    final_list = []
    for the_key, the_value in feature_dict.items():
        final_list.append(the_value.cpu())

    return final_list


def loader_to_tensor(args, loader):

    nr_iter = int(args.nrsamples / args.bs)

    final = []
    for it, data in enumerate(loader):
        final.append(data)

        if it >= nr_iter:
            break
    
    tensor = torch.vstack(final)

    return tensor


def get_image_names(pth):
    myimages = [] #list of image filenames
    dirFiles = os.listdir(pth) #list of directory files
    dirFiles = natsort.natsorted(dirFiles,reverse=False)

    for files in dirFiles: # filter out all non jpgs
        if '.png' in files:
            myimages.append(files)
            
    return myimages


def load_png(args, pth):
    import os, PIL.Image
    image_names = get_image_names(pth)[:args.nrsamples]
    
    loaded_images = []
    for image_name in image_names:
        cur_img = PIL.Image.open( os.path.join(pth, image_name) )
        cur_img = cur_img.convert('RGB')
        cur_img = np.asarray(cur_img, dtype=np.float32) / 255.
        loaded_images.append(cur_img.transpose((2,0,1)))

    stacked_images = torch.from_numpy(np.stack(loaded_images, axis=0))
    
    return stacked_images


def load_data(args, ws_gen_path):

    nor_img = None
    adv_img = None

    if '.pt' in args.load_gen_nor:
        nor_pth = os.path.join(ws_gen_path, args.dataset, args.load_gen_nor)
        nor_img = torch.load(nor_pth)[:args.nrsamples]
    if '.pt' in args.load_gen_adv:   
        adv_pth = os.path.join(ws_gen_path, args.dataset, args.load_gen_adv)
        adv_img = torch.load(adv_pth)[:args.nrsamples]


    if '_png' in args.load_gen_nor:
        nor_pth = os.path.join(ws_gen_path, args.dataset, args.load_gen_nor)
        nor_img = load_png(args, nor_pth)[:args.nrsamples]
    if '_png' in args.load_gen_adv:   
        adv_pth = os.path.join(ws_gen_path, args.dataset, args.load_gen_adv)
        adv_img = load_png(args, adv_pth)[:args.nrsamples]

    

    if args.dataset == 'stablediffv21_laion768':
        if '_jpg' in args.load_gen_nor or '_jpg' in args.load_gen_adv:
            from helper_gen.laion import get_laionhq_loader
            img_dir = os.path.join(results_path, "gen/stablediffv21_laion768")

            if "crop" in args.load_gen_nor:
                csv_path = os.path.join(results_path, "gen/stablediffv21_laion768/crop.txt")
            else: 
                csv_path = os.path.join(results_path, "gen/stablediffv21_laion768/border.txt")
            loader = get_laionhq_loader(args, csv_path, img_dir, data='images', img_size=763, batch_size=args.bs, num_workers=8, shuffle=False)
            nor_img = loader_to_tensor(args, loader)[:args.nrsamples]

            csv_path = os.path.join(results_path, "gen/stablediffv21_laion768/adv_prompt_jpg.txt")
            loader = get_laionhq_loader(args, csv_path, img_dir, data='images', img_size=763, batch_size=args.bs, num_workers=8, shuffle=False)
            adv_img = loader_to_tensor(args, loader)[:args.nrsamples]

            del loader

    elif args.dataset == 'diffusiondb512':
        if '_jpg' in args.load_gen_nor or '_jpg' in args.load_gen_adv:
            from helper_gen.laion import get_laionhq_loader
            
            img_dir = os.path.join(results_path, "gen", args.dataset)

            if "crop" in args.load_gen_nor:
                csv_path = os.path.join(results_path, "gen", args.dataset, "crop.txt")
            else: 
                print("notfound!")

            if 'sac512' in args.load_gen_nor:
                img_path = os.path.join(results_path, "gen", args.dataset, args.load_gen_nor)
                from helper_gen.sac import (  get_sac_imageloader )
                loader = get_sac_imageloader(args, img_path, data='images', batch_size=args.bs, num_workers=0)

            else:
                loader = get_laionhq_loader(args, csv_path, img_dir, data='images', img_size=512, batch_size=args.bs, num_workers=8, shuffle=False)

            nor_img = loader_to_tensor(args, loader)[:args.nrsamples]

            csv_path = os.path.join(results_path, "gen", args.dataset, "adv_large_random_5k_jpg.txt")
            loader = get_laionhq_loader(args, csv_path, img_dir, data='images', img_size=512, batch_size=args.bs, num_workers=8, shuffle=False)
            adv_img = loader_to_tensor(args, loader)[:args.nrsamples]

            del loader


    print("nor shape> ", nor_img.shape)
    print("adv shape> ", adv_img.shape)

    assert(nor_img.shape[0] == adv_img.shape[0])

    return nor_img, adv_img