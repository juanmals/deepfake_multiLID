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





class SACDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, data='images', transform=None):

        print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)
    
        self.df = pd.read_fwf(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path

        file1 = open(csv_path, 'r')
        Lines = file1.readlines()
        lines = [line.rstrip('\n') for line in Lines if line.strip() != '']

        self.img_names = lines
        self.transform = transform
        

    def __getitem__(self, index):
        
        pth = os.path.join(self.img_dir, self.img_names[index])
        # print("pth", pth)
        img = Image.open(pth)
        img = img.convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)

        return img


    def __len__(self):
        return self.df.shape[0]




class SACDatasetDir(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, img_dir, data='images', transform=None):

        print("img_dir: ", img_dir)

        import glob
        self.img_names = sorted(glob.glob(img_dir + '/'  + '*.jpg'))
        self.transform = transform


    def __getitem__(self, index):
   
        img = Image.open(self.img_names[index])
        img = img.convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)

        return img


    def __len__(self):
        return len(self.img_names)




def get_sac_trainingloader(args, img_dir, data='images', batch_size=64, num_workers=8, shuffle=True):

    train_transform = transforms.Compose([transforms.ToTensor()])
    
    img_dir

    train_dataset = SACDataset( 
                                csv_path=csv_path,
                                img_dir=img_dir,
                                data=data,
                                transform=train_transform
                            )

    train_loader = DataLoader(
                                dataset=train_dataset,
                                batch_size=batch_size,
                                # shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False,
                                # sampler=SubsetRandomSampler(torch.randint(high=len(train_dataset), size=(2000,)) )
                            )

    return train_loader



def get_sac_imageloader(args, img_dir, data='images', batch_size=64, num_workers=8, shuffle=True):

    train_transform = transforms.Compose([transforms.ToTensor()])
    

    train_dataset = SACDatasetDir( 
                                img_dir=img_dir,
                                data=data,
                                transform=train_transform
                            )

    train_loader = DataLoader(
                                dataset=train_dataset,
                                batch_size=batch_size,
                                # shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False,
                                # sampler=SubsetRandomSampler(torch.randint(high=len(train_dataset), size=(2000,)) )
                            )

    return train_loader



if __name__ == "__main__":
    import json
    import os, sys
    sys.path.insert(1, '/home/lorenzp/DeepfakeDetection')
    from helper_gen_images import calc_mean_std_without_target3

    from misc import ( create_dir )

    sys.path.insert(1, '/home/lorenzp/DeepfakeDetection')
    class Args():
        bs = 32

    args = Args()

    train_loader = get_sac_trainingloader(args, data='images', batch_size=args.bs, num_workers=0)

    total_mean, total_std = calc_mean_std_without_target3(train_loader)

    final_json = {"mean": total_mean.tolist(), "var": total_std.tolist()}

    outputdir = "/home/lorenzp/DeepfakeDetection/analysis/sac/std_mean_dev"
    create_dir(outputdir)
    output_file = os.path.join(outputdir, "all.json")
    print("dump json> ", output_file)
    with open(output_file, 'w') as json_file:
        json.dump(final_json, json_file)