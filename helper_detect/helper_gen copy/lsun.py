

import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset


def train_loader(args, dataset_id="pcuenq/lsun-bedrooms", image_size=256):
    from cfg import huggingface_cache_path

    # import pdb; pdb.set_trace()

    # huggingface_cache_path = "/"

    dataset = load_dataset(dataset_id, split="train", cache_dir=huggingface_cache_path)
    batch_size = args.bs

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            # https://www.kaggle.com/code/kuzand/testing-a-flower-classifier
            # transforms.Resize((image_size, image_size)),  # Resize
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # Convert to tensor (0, 1)
        ]
    )
    breakpoint()

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_dataloader


if __name__ == "__main__":
    import json
    import os, sys
    from helper_gen_images import calc_mean_std_without_target

    sys.path.insert(1, '/home/lorenzp/DeepfakeDetection')
    class Args():
        bs = 128

    args = Args()
    train_loader = train_loader(args, dataset_id="pcuenq/lsun-bedrooms", image_size=256)

    total_mean, total_std = calc_mean_std_without_target(train_loader)

    final_json = {"mean": total_mean.tolist(), "var": total_std.tolist()}

    outputdir = "/home/lorenzp/DeepfakeDetection/analysis/lsun/std_mean_dev"
    output_file = os.path.join(outputdir, "lsun-bedrooms.json")
    print("dump json> ", output_file)
    with open(output_file, 'w') as json_file:
        json.dump(final_json, json_file)