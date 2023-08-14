""" train and test dataset

author baiyu
"""
import os
import sys
import pdb

import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms


from cfg import *


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, data='Gender', transform=None):

        print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)
    
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = self.df.index.values
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


def get_celebaHQ_trainingloader(args, data='Hair_Color', img_size=256, batch_size=64, num_workers=8, shuffle=True):

    train_transform = transforms.Compose([transforms.ToTensor()])

    csv_path = "/home/lorenzp/celebA/celeba-train_hair_color_hq_ext_70.csv"
    img_dir = "/home/DATA/ITWM/lorenzp/CelebAHQ/Img/hq/data{}x{}".format(img_size, img_size)


    train_dataset = CelebaDataset(  csv_path=csv_path,
                                    img_dir=img_dir,
                                    data=data,
                                    transform=train_transform
                                    )


    train_loader = DataLoader(  dataset=train_dataset,
                                batch_size=batch_size,
                                # shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False,
                                sampler=SubsetRandomSampler(torch.randint(high=len(train_dataset), size=(2000,)) )
                                )

    return train_loader


def get_validation_dataloader(mean, std, data='Hair_Color', batch_size=64, num_workers=8, shuffle=False):

    csv_path = '../pytorch_ipynb/cnn/celeba-valid_' + data.lower() + '_hq_' + DATA_SPLIT + '.csv'

    val_transform = transforms.Compose(get_compose(mean, std))

    val_dataset = CelebaDataset(csv_path=csv_path,
                                img_dir=IMG_DIR,
                                data=data,
                                transform=val_transform)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=False)

    return val_loader

def get_test_dataloader(mean, std, data='Gender', batch_size=64, num_workers=8, shuffle=False):

    csv_path = '../pytorch_ipynb/cnn/celeba-test_' + data.lower() + '_hq_' + DATA_SPLIT + '.csv'

    test_transform = transforms.Compose(get_compose(mean, std))

    test_dataset = CelebaDataset(   csv_path=csv_path,
                                    img_dir=IMG_DIR,
                                    data=data,
                                    transform=test_transform)

    test_loader = DataLoader(   dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    return test_loader