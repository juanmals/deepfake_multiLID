
import torch
import torchvision
from torchvision import transforms
from datasets import load_dataset



def train_loader(args, image_size=32):
    from cfg import huggingface_cache_path
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train", cache_dir=huggingface_cache_path)

    # Define data augmentations
    preprocess = transforms.Compose(
        [
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
        batch_size=args.bs, 
        shuffle=False
    )

    return train_dataloader