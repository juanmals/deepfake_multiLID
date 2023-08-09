""" train and test dataset

author baiyu
"""
import os
import sys
import pdb
import numpy as np
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms

import json


class CiFakeDataset(Dataset):

    def __init__(self, csv_path, img_dir, data='labels', transform=transforms.Compose([transforms.ToTensor()])):

        print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)
    
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = self.df['images'].values
        self.y = self.df[data].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.df.shape[0]


def get_cifake_trainingloader(args, num_workers=8, shuffle=False):

    if args.mode == 'nor':
        pth = '/home/lorenzp/DeepfakeDetection/analysis/cifake/train_real.csv'
        img_dir = '/home/DATA/ITWM/lorenzp/cifake/train/REAL'
    else:
        pth = '/home/lorenzp/DeepfakeDetection/analysis/cifake/train_fake.csv'
        img_dir = '/home/DATA/ITWM/lorenzp/cifake/train/FAKE'

    tf = transforms.Compose([transforms.ToTensor()])

    dataset = CiFakeDataset(pth, img_dir, transform=tf)

    train_loader = DataLoader(  
                                dataset=dataset,
                                batch_size=args.bs,
                                shuffle=shuffle,
                                num_workers=8,
                                drop_last=False
                            )

    return train_loader


def generate_training_samples(args, shuffle=False):
    training_loader = get_cifake_trainingloader(args, num_workers=8, shuffle=shuffle)

    total = int(args.max_nr / args.bs)
    images = []
    for img, _  in tqdm(training_loader, total=total):
        images.append(img)

    batch = torch.vstack(images)

    return batch


if __name__ == "__main__":
    print("cald mean and dev!")
    from helper_gen_images import ( calc_mean_std )

    pth = '/home/lorenzp/DeepfakeDetection/analysis/cifake/train_real.csv'
    img_dir = '/home/DATA/ITWM/lorenzp/cifake/train/REAL'
    outputdir = "/home/lorenzp/DeepfakeDetection/analysis/cifake/std_mean_dev"

    tf = transforms.Compose([transforms.ToTensor()])

    dataset = CiFakeDataset(pth, img_dir, transform=tf)

    train_loader = DataLoader(  
                                dataset=dataset,
                                batch_size=1024,
                                shuffle=False,
                                num_workers=8,
                                drop_last=False
                            )

    total_mean, total_std = calc_mean_std(train_loader)

    final_json = {"mean": total_mean.tolist(), "var": total_std.tolist()}

    output_file = os.path.join(outputdir, "cifake.json")
    print("dump json> ", output_file)
    with open( output_file, 'w') as json_file:
        json.dump(final_json, json_file)