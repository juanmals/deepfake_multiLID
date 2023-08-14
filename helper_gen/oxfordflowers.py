import os
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset



def train_loader(args, image_size=32):
    from cfg import huggingface_cache_path
    dataset = load_dataset("huggan/flowers-102-categories", split="train", cache_dir=huggingface_cache_path)
    batch_size = args.bs

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            # https://www.kaggle.com/code/kuzand/testing-a-flower-classifier
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.ToTensor(),  # Convert to tensor (0, 1)
        ]
    )

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
    print("cald mean and dev!")
    import json
    from helper_gen_images import ( calc_mean_std_without_target )
    from misc import ( create_dir )

    dataset_path = "/home/DATA/ITWM/lorenzp"
    huggingface_cache_path = os.path.join(dataset_path, 'huggingface')

    dataset = load_dataset("huggan/flowers-102-categories", split="train", cache_dir=huggingface_cache_path)
    batch_size = 128
    image_size = 64

    # Define data augmentations
    preprocess = transforms.Compose(
        [   
            # https://www.kaggle.com/code/kuzand/testing-a-flower-classifier
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.ToTensor(),  # Convert to tensor (0, 1)
        ]
    )

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


    total_mean, total_std = calc_mean_std_without_target(train_dataloader)

    final_json = {"mean": total_mean.tolist(), "var": total_std.tolist()}

    outputdir = "/home/lorenzp/DeepfakeDetection/analysis/oxfordflowers/std_mean_dev"
    create_dir(output_dir)
    output_file = os.path.join(outputdir, "oxfordflowers64.json")
    print("dump json> ", output_file)
    with open(output_file, 'w') as json_file:
        json.dump(final_json, json_file)