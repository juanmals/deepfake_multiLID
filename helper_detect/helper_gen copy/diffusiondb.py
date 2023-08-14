import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset


def train_loader(args, dataset_id, image_size=512):
    from cfg import huggingface_cache_path
    dataset = load_dataset('poloclub/diffusiondb', dataset_id,  split="train", cache_dir=huggingface_cache_path)

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.CenterCrop((image_size, image_size)),  # Resize https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html
            transforms.ToTensor(),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform) 


    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.bs, 
        shuffle=False
    )

    return train_dataloader
