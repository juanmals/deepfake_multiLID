import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from datasets import load_dataset
from urllib.request import urlopen
from PIL import Image
import requests
from tqdm import tqdm
import logging
import time
import pandas as pd
import io


# from helper_gen.helper_gen_images import ( create_save_dir )

def load_hf_dataset(args, dataset_id):
    from cfg import huggingface_cache_path
    dataset = load_dataset(dataset_id, split="train", cache_dir=huggingface_cache_path)

    return dataset


def save_csv(save_dir, dictionary):

    df = pd.DataFrame(dictionary)
    df.to_csv(os.path.join(save_dir, "alog.csv"))


def save_kaionHQ_dataset(args, dataset, image_size):

    max_samples = dataset.num_rows
    save_dir = create_save_dir(args)

    saved_urls = []
    saved_iters = []
    saved_text = []

    it = 0

    while True:
        try:
            image_url = dataset[it]['URL']
            it = it + 1

            if len(image_url) == 0:
                print("Err> Url len == 0")
                continue

        
            req = requests.get(image_url, stream=True)
            bytes_cont = io.BytesIO(req.content)
            bytes_cont.seek(0)
            im = Image.open(bytes_cont)
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")

            resized = im.resize((image_size, image_size))

            resized.save(os.path.join(save_dir, "{}.jpg".format(it)))
            # logging.info('yes: ', image_url)
            # breakpoint()
            # time.sleep( 20 / 1000)
            saved_urls. append(image_url)
            saved_iters.append(it)
            saved_text.append(dataset[it]['TEXT'])
            final = {'ITER': saved_iters, 'URL': saved_urls, 'TEXT': saved_text}

            if it % int(args.max_nr / 100) == 0:
                tqdm(len(saved_urls), total=args.max_nr)
                save_csv(save_dir, final)

            if len(saved_urls) >= args.max_nr:
                save_csv(save_dir, final)
                break

        except Exception as er:
            print("An exception occurred", it, image_url)
            print(er)
            continue


class LaionHQDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, data='images', transform=None):

        print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)
    
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = self.df[data].values
        # self.y = self.df[data].values
        self.transform = transform
        

    def __getitem__(self, index):
        
        pth = os.path.join(self.img_dir, self.img_names[index])
        # print("pth", pth)
        img = Image.open(pth)
        img = img.convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        # label = 0 # self.y[index]

        return img


    def __len__(self):
        return self.df.shape[0]


def get_laionhq_trainingloader(args, data='images', img_size=763, batch_size=64, num_workers=8, shuffle=True):

    train_transform = transforms.Compose([transforms.ToTensor()])

    csv_path = "/home/lorenzp/workspace/DeepfakeDetection/results/gen/stablediffv21_laion768/crop.txt"
    img_dir = "/home/lorenzp/workspace/DeepfakeDetection/results/gen/stablediffv21_laion768"

    train_dataset = LaionHQDataset( 
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


def get_laionhq_loader(args, csv_path, img_dir, data='images', img_size=763, batch_size=64, num_workers=8, shuffle=True):

    train_transform = transforms.Compose([transforms.ToTensor()])


    train_dataset = LaionHQDataset( 
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

    train_loader = get_laionhq_trainingloader(args, data='images', img_size=763, batch_size=args.bs, num_workers=0, shuffle=False)

    total_mean, total_std = calc_mean_std_without_target3(train_loader)

    final_json = {"mean": total_mean.tolist(), "var": total_std.tolist()}

    outputdir = "/home/lorenzp/DeepfakeDetection/analysis/stablediffv21/std_mean_dev"
    create_dir(outputdir)
    output_file = os.path.join(outputdir, "crop.json")
    print("dump json> ", output_file)
    with open(output_file, 'w') as json_file:
        json.dump(final_json, json_file)