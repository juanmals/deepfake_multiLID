import os
import json
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

# import detectors
import timm

from cfg import *


def get_normalization(args):
    mean = None
    std  = None
    if args.dataset in ['cifar10']:
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
    elif args.dataset in ['cifake']:
        mean = [0.4913, 0.4821, 0.4465]
        std  = [0.2457, 0.2430, 0.2611]
    elif args.dataset in ['mnist']:
        mean = [0.1307, 0.1307, 0.1307]
        std  = [0.3081, 0.3081, 0.3081]
    elif args.dataset in ['cifar100']:
        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2565, 0.2761]
    elif args.dataset in ['butterflies32']:
        mean = [0.7687, 0.7198, 0.6324]
        std  = [0.2675, 0.2940, 0.3371]
    elif args.dataset in ['oxfordflowers64']: # https://github.com/Muhammad-MujtabaSaeed/102-Flowers-Classification/blob/master/102_Flowers_classification.ipynb
        mean = [0.4355, 0.3777, 0.2880]
        std  = [0.2855, 0.2331, 0.2586]
    elif args.dataset in ['butterflies128']: 
        mean = [0.7687, 0.7198, 0.6324]
        std  = [0.2675, 0.2940, 0.3371]
    elif args.dataset in ['celebaHQ256']:
        mean = [0.5173, 0.4169, 0.3636]
        std  = [0.3028, 0.2742, 0.2691]
    elif args.dataset in ['celebaHQ128']:
        mean = [0.5175, 0.4170, 0.3637]
        std =  [0.3028, 0.2741, 0.2690]
    elif args.dataset in ['lsun-bedroom256']: 
        mean = [0.4804, 0.4508, 0.4002]
        std  = [0.2748, 0.2666, 0.2784]
    elif args.dataset in ['lsun-cat256']: 
        mean = [0.4804, 0.4508, 0.4002]
        std  = [0.2748, 0.2666, 0.2784]
    elif args.dataset in ['lsun-church256']: 
        mean = [0.4804, 0.4508, 0.4002]
        std  = [0.2748, 0.2666, 0.2784]
    elif args.dataset in ['ffhq256']: 
        mean = [0.5209, 0.4257, 0.3811]
        std  = [0.2826, 0.2572, 0.2576]
    elif args.dataset in ['ffhq1024']: 
        mean = [0.5209, 0.4257, 0.3811]
        std  = [0.2826, 0.2572, 0.2576]
    elif args.dataset in ['imagenet224']:
        mean = [0.4804, 0.4508, 0.4002]
        std  = [0.2748, 0.2666, 0.2784]
    elif args.dataset in ['imagenet']:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    elif args.dataset in ['af-all']:
        mean = [0.4851, 0.4507, 0.4148]
        std  = [0.2797, 0.2714, 0.2829]
    elif args.dataset in ['af-afhq']:
        mean = [0.5058, 0.4573, 0.3989]
        std  = [0.2516, 0.2412, 0.2441]
    elif args.dataset in ['af-celebahq']:
        mean = [0.5468, 0.4263, 0.3645]
        std  = [0.2953, 0.2596, 0.2503]
    elif args.dataset in ['af-coco']:
        mean = [0.4750, 0.4467, 0.4058]
        std  = [0.2721, 0.2667, 0.2803]
    elif args.dataset in ['af-imagenet']:
        mean = [0.4804, 0.4508, 0.4002]
        std  = [0.2748, 0.2666, 0.2784]
    elif args.dataset  in ['af-lsun', 'af-church', 'af-bedroom']:
        mean = [0.4757, 0.4541, 0.4263]
        std  = [0.2829, 0.2798, 0.2925]  
    elif args.dataset in ['af-metfaces']:
        mean = [0.4820, 0.4092, 0.3413]
        std  = [0.2490, 0.2299, 0.2229]
    elif args.dataset in ['af-cycle_gan']:
        mean = [0.4823, 0.4726, 0.4345]
        std  = [0.2478, 0.2363, 0.2553]
    elif args.dataset in ['stablediffv21_laion768']:
        mean = [0.6341, 0.6090, 0.5891] # 8000 samples
        std  = [0.3356, 0.3368, 0.3478]
    elif args.dataset in ['diffusiondb512']:  # like laion5b
        mean = [0.6341, 0.6090, 0.5891]
        std  = [0.3356, 0.3368, 0.3478]
    elif args.dataset in ['sac512']:  
        mean = [0.4674, 0.4328, 0.4063]
        std  = [0.2732, 0.2653, 0.2599]
    else:
        raise NotImplementedError("Mean and standard not implemented!")
    return mean, std


def normalize_images(args, images):
    mean, std = get_normalization(args)
    images[:,0,:,:] = (images[:,0,:,:] - mean[0]) / std[0]
    images[:,1,:,:] = (images[:,1,:,:] - mean[1]) / std[1]
    images[:,2,:,:] = (images[:,2,:,:] - mean[2]) / std[2]
    return images


def check_file_ending(path, ending='.json'):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '':
        path = path + ending
    return path


def load_args(args, parser):
    print("load json>", args.load_json)
    filename = check_file_ending(args.load_json)
    with open(filename, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    return args


def save_args(args):
    # Optional: support for saving settings into a json file
    print("save json>", args.save_json)
    filename = check_file_ending(args.save_json)
    if filename:
        with open(filename, 'wt') as f:
            json_args = vars(args)
            del json_args['save_json']
            json.dump(json_args, f, indent=4)


def args_handling(args, parser, cfg_path):
    args.load_json = os.path.join(cfg_path, args.load_json)
    if not args.save_json == "":
        save_args(args)
    else:
        args = load_args(args, parser)
    return args


def print_args(args):
    print(''.join(f'{k}={v} \n' for k, v in vars(args).items()))


def create_dir(path):
    is_existing = os.path.exists(path)
    if not is_existing:
        os.makedirs(path)
        print("The new directory is created!", path)


def check_str_startswith(string, substring):
    start = string.split('_')[0]
    if start in substring:
        return True
    return False


def create_pth(args, ws_path, filename, dataset, join=True):
    save_dir = os.path.join(ws_path, dataset)
    if ws_path == ws_extract_path:
        save_dir = os.path.join(ws_path, dataset, args.extract)
    
    if join:
        to_save = os.path.join(save_dir, filename)
        print("save> ",  to_save)
        return to_save

    return save_dir, filename


def save_to_pt(args, ws_path, payload, filename):
    savedir, filename = create_pth(args, ws_path, filename, args.dataset, join=False)
    create_dir(savedir)
    pth_filename = os.path.join(savedir, filename)
    print("save to> ", pth_filename)
    
    torch.save(payload,  pth_filename)


def remove_data_parallel(old_state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict


def load_model_timm(args):
    if args.model == 'rn18':
        pretrained = True if not args.pretrained == "" else False

        # https://huggingface.co/edadaltocg/resnet18_cifar10
        model = timm.create_model("resnet18", num_classes=10, pretrained=False)
        # override model
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()  # type: ignore
        # model.fc = nn.Linear(512,  10)

        if args.dataset == 'cifar10' and pretrained:
            model.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                                "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
                                map_location="cpu", 
                                file_name="resnet18_cifar10.pth",
                        )
            )

    elif args.dataset == 'celebaHQ256' and args.model == 'rn18':
        model = timm.create_model("resnet18", num_classes=4, pretrained=False)
        if not args.pretrained == "":
            checkpoint = torch.load( os.path.join(ckpt_path, args.pretrained) )
            try:
                model.load_state_dict(checkpoint['net'])
            except:
                ckpt = remove_data_parallel(checkpoint['net'])
                model.load_state_dict(ckpt)
            
            best_acc = checkpoint['acc']
            print("checkpoint loaded ...")
            print("best acc> ", best_acc)

    model.eval()
    model.cuda()
    cudnn.benchmark = True
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

    return model 


def load_model(args):
    if args.dataset == 'cifar10' and args.model == 'rn18':
        from models import resnet
        model = resnet.ResNet18()

    model.eval()
    model.cuda()
    cudnn.benchmark = True

    if not args.pretrained == "":
        if args.model == 'rn18':
            checkpoint = torch.load( os.path.join(ckpt_path, args.pretrained) )
            # model = torch.nn.DataParallel(model)
            try:
                model.load_state_dict(checkpoint['net'])
            except:
                ckpt = remove_data_parallel(checkpoint['net'])
                model.load_state_dict(ckpt)
            
            best_acc = checkpoint['acc']
            print("checkpoint loaded ...")
            print("best acc> ", best_acc)

    return model


def create_log_file(args):    
    create_dir(log_path)
    log = vars(args)
    return log


def save_log(args, log_dict, load_cfg):
    save_dir = os.path.join(log_path, load_cfg)
    print("save log ...", save_dir)
    print(log_dict)

    create_dir(os.path.dirname(save_dir))

    with open(save_dir, "w") as write_file:
        try:
            json.dump(log_dict, write_file, indent=4)
        finally:
            write_file.close()
