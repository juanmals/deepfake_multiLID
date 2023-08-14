
import os
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from datasets import load_dataset


def loat_dataset(args, IS_TRAIN=True, shuffle=True):

    IMAGENET_PATH      = "/home/DATA/ITWM/lorenzp/ImageNet"
    num_workers = 8

    transform_list = [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()] 
    transform = transforms.Compose(transform_list)

    dataset_dir = os.path.join(IMAGENET_PATH, 'val') if IS_TRAIN else os.path.join(IMAGENET_PATH, 'train')
    data_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dir, transform), batch_size=args.bs, shuffle=shuffle, 
                        num_workers=num_workers, pin_memory=True)

    return data_loader
