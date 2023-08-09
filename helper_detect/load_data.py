import torch
from cfg import *

from misc import (
    args_handling,
    print_args,
    create_dir,
    save_to_pt,
    create_pth,
    create_log_file,
    save_log
)


def load_data(args, train_samples, dataset, transfer=False):
    # if args.dataset in ['cifar10', 'mnist', 'butterfly32', 'celebaHQ256']:
    if args.extract == 'normal':
        adv = torch.load("./pytorch-ddpm-cifar10/data/samples2000/ddpm_ema.pt")[:args.nrsamples]
        nor = torch.load("./denoising-diffusion-pytorch/data/cifar10_train.pt")[:args.nrsamples]
    
    elif args.extract == 'features':
        args.mode='nor'; nor_pth = create_pth(args, ws_extract_path, filename=args.load_extr_nor if not transfer else args.load_extr_nor_trans, dataset=dataset)
        args.mode='adv'; adv_pth = create_pth(args, ws_extract_path, filename=args.load_extr_adv if not transfer else args.load_extr_adv_trans, dataset=dataset)

        print("open file> ", nor_pth)
        print("open file> ", adv_pth)

        nor = torch.load(nor_pth)
        adv = torch.load(adv_pth)

        adv = adv[1]['1_conv2_1'].cpu().numpy()[:args.nrsamples]
        nor = nor[1]['1_conv2_1'].cpu().numpy()[:args.nrsamples]
        
    elif args.extract == 'fft':
        adv = torch.load("./denoising-diffusion-pytorch/data/extract/cifar10_fft_adv.pt")[:args.nrsamples]
        nor = torch.load("./denoising-diffusion-pytorch/data/extract/cifar10_fft_ema.pt")[:args.nrsamples]

    elif args.extract in ['bb-multiLID', 'wb-multiLID', 'LID']:
        args.mode='nor'; nor_pth = create_pth(args, ws_extract_path, filename=args.load_extr_nor if not transfer else args.load_extr_nor_trans, dataset=dataset)
        args.mode='adv'; adv_pth = create_pth(args, ws_extract_path, filename=args.load_extr_adv if not transfer else args.load_extr_adv_trans, dataset=dataset)

        print("open file> ", nor_pth)
        print("open file> ", adv_pth)

        nor = torch.load(nor_pth)[:args.nrsamples]
        adv = torch.load(adv_pth)[:args.nrsamples]

        print("adv.shape: ", adv.shape)
        nor = nor.reshape((nor.shape[0], -1))
        adv = adv.reshape((adv.shape[0], -1))
    # else:
    #     raise NotImplementedError("dataset not found!")

    return nor, adv


def load_data_adv(args, train_samples, dataset, load_extr_adv):
    if args.extract == 'normal':
        nor = torch.load("./denoising-diffusion-pytorch/data/cifar10_train.pt")[:args.nrsamples]
    
    elif args.extract == 'features':
        args.mode='nor'; nor_pth = create_pth(args, ws_extract_path, filename=args.load_extr_nor if not transfer else args.load_extr_nor_trans, dataset=dataset)
        args.mode='adv'; adv_pth = create_pth(args, ws_extract_path, filename=args.load_extr_adv if not transfer else args.load_extr_adv_trans, dataset=dataset)

        print("open file> ", nor_pth)
        print("open file> ", adv_pth)

        nor = torch.load(nor_pth)
        adv = torch.load(adv_pth)

        adv = adv[1]['1_conv2_1'].cpu().numpy()[:args.nrsamples]
        nor = nor[1]['1_conv2_1'].cpu().numpy()[:args.nrsamples]
        
    elif args.extract == 'fft':
        adv = torch.load("./denoising-diffusion-pytorch/data/extract/cifar10_fft_adv.pt")[:args.nrsamples]
        nor = torch.load("./denoising-diffusion-pytorch/data/extract/cifar10_fft_ema.pt")[:args.nrsamples]

    elif args.extract in ['bb-multiLID', 'wb-multiLID', 'LID']:
        args.mode='adv'; adv_pth = create_pth(args, ws_extract_path, filename=load_extr_adv, dataset=dataset)

        print("open file> ", adv_pth)
        adv = torch.load(adv_pth)
        print("adv.shape: ", adv.shape)
        adv = adv.reshape((adv.shape[0], -1))


    return adv