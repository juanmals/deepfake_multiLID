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
import json

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms

from cfg import *


class ArtiFactDataset(Dataset):

    def __init__(self, df, img_dir, transform=transforms.Compose([transforms.ToTensor()])):
        # print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)
        self.df = df
        self.img_dir = img_dir
        self.img_names = self.df['image_path'].values
        self.y = self.df['target'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.df.shape[0]


def af_ddpm_trainingloader(args, category, num_workers=8, shuffle=False):

    base_dir = '/home/DATA/ITWM/lorenzp/artifact'
    if args.mode == 'nor':
        tf = transforms.Compose([transforms.ToTensor()])
        pth = 'lsun'
    else:
        tf = transforms.Compose([transforms.ToTensor()])
        pth = 'ddpm'

    df = pd.read_csv(os.path.join(base_dir, pth, "metadata.csv"), index_col=0)
    df = df[df["category"].eq(category)]
    model_dir = os.path.join(base_dir, pth)
    dataset = ArtiFactDataset(df,  model_dir, transform=tf)

    train_loader = DataLoader(  
                                dataset=dataset,
                                batch_size=args.bs,
                                shuffle=shuffle,
                                num_workers=8,
                                drop_last=False
                            )

    return train_loader


def find_path(image_path):
    tmp = ""
    if "afhq/" in image_path:
        tmp = "afhq"
    elif "big/" in image_path:
        tmp = "big_gan"
    elif "celebahq/data" in image_path:
        tmp = "celebahq"
    elif "cips/" in image_path:
        tmp = "cips"
    elif "coco2017/" in image_path:
        tmp = "coco"
    elif "st/" in image_path:
        tmp = "cycle_gan"
    elif "ddpm/" in image_path:
        tmp = "ddpm"
    elif "denoising-diffusion-gan-data/" in image_path:
        tmp = "denoising_diffusion_gan"
    elif "/Diffusion-ProjectedGAN" in image_path:
        tmp = "diffusion_gan"
    elif "facesyn/" in image_path:
        tmp = "face_synthetics"
    elif "ffhq/images" in image_path:
        tmp = "ffhq"
    elif "gf/" in image_path:
        tmp = "gansformer" 
    elif "gau/" in image_path:
        tmp = "gau_gan" 
    elif "gc-in/" in image_path:
        tmp = "generative_inpainting" 
    elif "gcinpaint/" in image_path:
        tmp = "generative_inpainting" 
    elif "glide/" in image_path:
        tmp = "glide" 
    elif "glide-in/" in image_path:
        tmp = "glide" 
    elif "glide-t2i/" in image_path:
        tmp = "glide" 
    elif "imagenet/" in image_path:
        tmp = "imagenet" 
    elif "lama/" in image_path:
        tmp = "lama" 
    elif "landscape/images" in image_path:
        tmp = "landscape" 
    elif "latentdiff/" in image_path:
        tmp = "latent_diffusion" 
    elif "latentdiff-t2i/" in image_path:
        tmp = "latent_diffusion"
    elif "bedroom/bedroom" in image_path:
        tmp = "lsun" 
    elif "car/car" in image_path:
        tmp = "lsun"
    elif "cat/cat" in image_path:
        tmp = "lsun" 
    elif "church/church" in image_path:
        tmp = "lsun"
    elif "horse/horse" in image_path:
        tmp = "lsun" 
    elif "mat/" in image_path:
        tmp = "mat"
    elif "metfaces/" in image_path:
        tmp = "metfaces" 
    elif "palette/" in image_path:
        tmp = "palette"
    elif "pro/" in image_path:
        tmp = "pro_gan"
    elif "proj/art_painting" in image_path:
        tmp = "projected_gan"
    elif "proj/bedroom" in image_path:
        tmp = "projected_gan"
    elif "proj/church" in image_path:
        tmp = "projected_gan"
    elif "proj/cityscapes" in image_path:
        tmp = "projected_gan"
    elif "proj/ffhq" in image_path:
        tmp = "projected_gan"
    elif "proj/landscape" in image_path:
        tmp = "projected_gan" 
    elif "sfhq/" in image_path:
        tmp = "sfhq"
    elif "stable/" in image_path:
        tmp = "stable_diffusion"
    elif "stable-face/" in image_path:
        tmp = "stable_diffusion"
    elif "star/" in image_path:
        tmp = "star_gan"
    elif "sg1/" in image_path:
        tmp = "stylegan1"
    elif "car-part1/" in image_path:
        tmp = "stylegan2"
    elif "car-part2/" in image_path:
        tmp = "stylegan2"
    elif "cat-part1/" in image_path:
        tmp = "stylegan2"
    elif "cat-part2/" in image_path:
        tmp = "stylegan2"
    elif "church-part1/" in image_path:
        tmp = "stylegan2"
    elif "church-part2/" in image_path:
        tmp = "stylegan2"
    elif "ffhq-part1/" in image_path:
        tmp = "stylegan2"
    elif "ffhq-part2/" in image_path:
        tmp = "stylegan2"
    elif "horse-part1/" in image_path:
        tmp = "stylegan2"
    elif "horse-part2/" in image_path:
        tmp = "stylegan2"
    elif "sg3/" in image_path:
        tmp = "stylegan3"
    elif "stylegan3/" in image_path:
        tmp = "stylegan3"
    elif "tt-cc/" in image_path:
        tmp = "taming_transformer"
    elif "tt-coco/" in image_path:
        tmp = "taming_transformer"
    elif "tt-ffhq/" in image_path:
        tmp = "taming_transformer"
    elif "vqdiff-imgnet/" in image_path:
        tmp = "vq_diffusion"
    else:
        raise NotImplementedError("Cannot parse: ", image_path)

    return tmp


class ArtiFactDatasetAll(Dataset):
    def __init__(self, df, img_dir, transform=transforms.Compose([transforms.ToTensor()])):
        # print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)
        self.df = df
        self.img_dir = img_dir
        self.img_names = self.df['image_path'].values
        self.y = self.df['target'].values
        self.transform = transform

    def __getitem__(self, index):
        dataset = find_path(self.img_names[index])
        img = Image.open(os.path.join(self.img_dir, dataset, self.img_names[index] ))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]

        return img, label

    def __len__(self):
        return self.df.shape[0]


def af_all_trainingloader(args, num_workers=8, shuffle=False):
    base_dir = '/home/lorenzp/DeepfakeDetection/analysis/artifact'
    tf = transforms.Compose([transforms.ToTensor()])
    if args.mode == 'nor':
        if "all_real2000" in args.save_gen_nor:
            csv = "all_real2000.csv"
        elif "all_real10000" in args.save_gen_nor:
            csv = "all_real10000.csv"
        elif "all_10000_balanced_csv" in args.save_gen_nor:
            csv = "all_real10000_target_3classes_balanced.csv"
        elif "all_10500_balanced_csv" in args.save_gen_nor:
            csv = "all_real10500_target_3classes_balanced.csv"
    else: 
        if "all_fake2000" in args.save_gen_adv:
            csv = "all_fake2000.csv"
        elif "all_fake10000" in args.save_gen_adv:
            csv = "all_fake10000.csv"
        elif "all_10000_diff_balanced_csv" in args.save_gen_adv:
            csv = "all_diff10000_target_3classes_balanced.csv"
        elif "all_10000_gan_balanced_csv" in args.save_gen_adv:
            csv = "all_gan10000_target_3classes_balanced.csv"
        elif "all_10500_fake_balanced_csv" in args.save_gen_adv:
            csv = "all_fake10500_target_3classes_balanced.csv"
    
    print("csv: ", csv)

    df = pd.read_csv(os.path.join(base_dir, csv), index_col=0)
    img_dir = "/home/DATA/ITWM/lorenzp/artifact"
    dataset = ArtiFactDatasetAll(df, img_dir, transform=tf)

    train_loader = DataLoader(  
                                dataset=dataset,
                                batch_size=args.bs,
                                shuffle=shuffle,
                                num_workers=8,
                                drop_last=False
                            )
    return train_loader


def gen_batch(args, training_loader, with_target=False):
    total = int(args.max_nr / args.bs)
    images = []
    for it, (img, target)  in tqdm(enumerate(training_loader), total=total):
        if with_target:
            images.append((img, target))
        else:
            images.append(img)
        if it * args.bs >= args.max_nr:
            break

    batch = torch.vstack(images)
    return batch


def gen_batch_target(args, training_loader):
    total = int(args.max_nr / args.bs)
    images = []
    for it, (img,target)  in tqdm(enumerate(training_loader), total=total):
        images.append(target)
        if it * args.bs >= args.max_nr:
            break
    batch = torch.stack(images)
    return batch


def generate_training_samples(args, category, shuffle=False):
    training_loader = af_ddpm_trainingloader(args, category, num_workers=8, shuffle=shuffle)
    batch = gen_batch(args, training_loader)
    return batch


def generate_training_samples_all(args, shuffle=False):
    training_loader = af_all_trainingloader(args, num_workers=8, shuffle=shuffle)
    batch = gen_batch(args, training_loader)
    return batch


def parse_df(df, dataset):

    # df = df[df["category"].eq(category)]
    if dataset == 'afhq':
        df = df[df["image_path"].str.contains("train")]
    elif dataset == 'coco':
        df = df[df["image_path"].str.contains("train")]
    elif dataset == 'imagnet':
        df = df[df["image_path"].str.contains("train")]

    return df


if __name__ == "__main__":

    from helper_gen_images import ( calc_mean_std )

    print("Calculate: std mean and dev!")
    base_dir = "/home/DATA/ITWM/lorenzp/artifact"
    meta = "metadata.csv"

    outputdir = "/home/lorenzp/DeepfakeDetection/analysis/artifact/std_mean_dev"

    all_true_csv = [
        "afhq",
        "celebahq",
        "coco",
        "ffhq",
        "imagenet", 
        "landscape",
        "lsun",
        "metfaces",
        "cycle_gan"
    ]

    categories = {
        "afhq": ["train"],
    }


    tf = transforms.Compose([transforms.ToTensor()])
    final_json = {}

    allcombined = True
    if allcombined:
        df_concat = []
        for dataset_name in all_true_csv:
            df = pd.read_csv(os.path.join(base_dir, dataset_name, meta), index_col=0)
            df = parse_df(df, dataset_name)
            df_concat.append(df)

        all_df = pd.concat(df_concat)


        img_dir = "/home/DATA/ITWM/lorenzp/artifact"
        dataset = ArtiFactDatasetAll(all_df, img_dir, transform=tf)

        image_loader = DataLoader(  
                                    dataset=dataset,
                                    batch_size=128,
                                    shuffle=False,
                                    num_workers=8,
                                    drop_last=False
                                )


        total_mean, total_std = calc_mean_std(image_loader)

        final_json = {"mean": total_mean.tolist(), "var": total_std.tolist()}

        output_file = os.path.join(outputdir, "af-all.json")
        print("dump json> ", output_file)
        with open( output_file, 'w') as json_file:
            json.dump(final_json, json_file)

    else:
        for dataset_name in all_true_csv:
            df = pd.read_csv(os.path.join(base_dir, dataset_name, meta), index_col=0)
            df = parse_df(df, dataset_name)
            
            img_dir = os.path.join(base_dir, dataset_name)

            dataset = ArtiFactDataset(df, img_dir, transform=tf)

            image_loader = DataLoader(  
                                        dataset=dataset,
                                        batch_size=128,
                                        shuffle=False,
                                        num_workers=8,
                                        drop_last=False
                                    )

            total_mean, total_std = calc_mean_std(image_loader)

            final_json[dataset_name] = {"mean": total_mean.tolist(), "var": total_std.tolist()}

            if dataset_name == all_true_csv[-1]:
                output_file = os.path.join(outputdir, "all" + ".json")
                print("dump json> ", output_file)
                with open( output_file, 'w') as json_file:
                    json.dump(final_json, json_file)

            output_file = os.path.join(outputdir, dataset_name + ".json")
            print("dump json> ", output_file)
            with open( output_file, 'w') as json_file:
                json.dump(final_json[dataset_name], json_file)

            del image_loader