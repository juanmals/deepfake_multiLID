import numpy as np
from tqdm import tqdm
import torch
import matplotlib.image as mpimg
from PIL import Image
from torchvision import transforms

from misc import (
    save_to_pt,
    create_pth,
    create_dir
)

from cfg import *


def check_fileformat(args):

    if hasattr(args, "fileformat") and len(args.fileformat) > 0:
        fileformat = args.fileformat
    else:
        fileformat = ['.pt']

    return fileformat


def create_save_dir(args):
    filename = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
    pth, _ = create_pth(args, ws_gen_path, filename, args.dataset, join=False)

    fileformat = check_fileformat(args)

    save_dirs = []
    for ff in fileformat:
        if '.png' == ff:
            tmp = filename.replace('.pt', '_png')
        
        if '.jpg' == ff:
            tmp = filename.replace('.pt', '_jpg')

        if not '.pt' == ff:
            save_dir = os.path.join(pth, tmp)
            create_dir(save_dir)
            save_dirs.append(save_dir)

    if len(save_dirs) == 0:
        raise ImplementationError("Fileformat not found!")

    return save_dirs


def save_images(args, image, it,  fileformat, save_dirs):
    if '.png' in fileformat:
        pth = os.path.join(save_dirs[0], "{:06}.png".format(it))
    elif '.jpg' in fileformat:
        pth = os.path.join(save_dirs[0], "{:06}.jpg".format(it))
    else:
        raise NotImplementedError("No fileformat found!")
    try:
        image.save(pth)
    except:
        print("Err: could not save image! at iter: ", it)
        

def generate_images(args, pipeline):
    """
    huggingface: pipeline
    """
    generated_images = []
    it = 0 
    save_dirs = create_save_dir(args)
    for _ in tqdm(range(int(args.max_nr / args.bs))):
        images = pipeline(batch_size=args.bs, output_type='np').images

        fileformat = check_fileformat(args)                

        if '.pt' in fileformat:
            generated_images.append(images)
        else:
            if images.shape[0] == 1:
                images = np.squeeze(images)
            images = Image.fromarray((images*255).astype(np.uint8))
            save_images(args, images, it, fileformat, save_dirs)

        it = it + 1

    if '.pt' in fileformat:
        batch = torch.from_numpy(np.transpose(np.vstack(generated_images), (0,3,1,2)))
        save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
        save_to_pt(args, ws_gen_path, batch, args.save_gen_adv)


def generate_prompt_images(args, pipeline, prompt="", resize=0, kwargs_tmp={}):
    """
    huggingface: pipeline. text2image. stable diffusion
    """
    fileformat = check_fileformat(args)
    generated_images = []

    save_dirs = create_save_dir(args)

    if prompt == "":
        from helper_gen.laion import load_hf_dataset
        dataset_id = "laion/laion-high-resolution"
        dataset = load_hf_dataset(args, dataset_id)
    
    for prompt_it in tqdm(range(args.max_nr), total=args.max_nr):
        if prompt == "":
            prompt = dataset[prompt_it]['TEXT']
        
        # images = pipeline(prompt, output_type='np').images
        if not len(kwargs_tmp) > 0:
            images = pipeline(prompt).images[0]
        else:
            images = pipeline([prompt], **kwargs_tmp).images[0]

        if resize > 0:
            images = images.resize((resize, resize), Image.BICUBIC)

        if ".pt" in fileformat:
            tmp = np.array(images, dtype='float32') / 255.
            generated_images.append(tmp)
        if ".png" in fileformat:
            save_images(args, images, prompt_it, [".png"], save_dirs)
        if ".jpg" in fileformat:
            save_images(args, images, prompt_it, [".jpg"], save_dirs)
        

    if '.pt' in fileformat:      
        batch = torch.from_numpy(np.transpose(np.vstack(generated_images), (0,3,1,2)))
        save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
        save_to_pt(args, ws_gen_path, batch, args.save_gen_adv)



def generate_images_from_hf(args, train_loader):
    """
    huggingface: train_loader. to image pt or jpg
    """
    fileformat = check_fileformat(args)
    generated_images = []
    # trans_pil_img = transforms.ToPILImage()

    save_dirs = create_save_dir(args)
    
    for it, img in tqdm(enumerate(train_loader), total=args.max_nr):
        
        # breakpoint()

        if ".pt" in fileformat:
            tmp = np.array(img['images'][0].numpy(), dtype='float32')
            generated_images.append(tmp)

        if args.dataset == 'sac512':
            pil_img = Image.fromarray(np.array(img[0].numpy().transpose((1,2,0))*255, dtype='uint8'), 'RGB')
        else:
            pil_img = Image.fromarray(np.array(img['images'][0].numpy().transpose((1,2,0))*255, dtype='uint8'), 'RGB')

        if ".png" in fileformat:
            save_images(args, pil_img, it, [".png"], save_dirs)
        if ".jpg" in fileformat:
            save_images(args, pil_img, it, [".jpg"], save_dirs)

    if '.pt' in fileformat:      
        batch = torch.from_numpy(np.transpose(np.vstack(generated_images), (0,3,1,2)))
        save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
        save_to_pt(args, ws_gen_path, batch, args.save_gen_adv)


def generate_images_from_loader(args, train_loader):
    """
    train_loader. to image pt or jpg or png
    """

    assert args.bs == 1

    fileformat = check_fileformat(args)
    generated_images = []
    trans_pil_img = transforms.ToPILImage('RGB')

    save_dirs = create_save_dir(args)

    iterations = int(args.max_nr / args.bs)
    for it, (img, target) in tqdm(enumerate(train_loader), total=iterations):
        
        if it  >= args.max_nr:
            break

        pil_img = trans_pil_img(img[0])

        if ".pt" in fileformat:
            tmp = np.array(img[0], dtype='float32')
            generated_images.append(tmp)

        if ".png" in fileformat:
            save_images(args, pil_img, it, [".png"], save_dirs)
        if ".jpg" in fileformat:
            save_images(args, pil_img, it, [".jpg"], save_dirs)

    if '.pt' in fileformat:      
        batch = torch.from_numpy(np.transpose(np.vstack(generated_images), (0,3,1,2)))
        save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
        save_to_pt(args, ws_gen_path, batch, args.save_gen_adv)



def generate_images_from_dit(args, class_ids, pipe, generator):
    """
    train_loader. to image pt or jpg or png
    """
    assert args.bs == 1

    fileformat = check_fileformat(args)
    generated_images = []
    trans_pil_img = transforms.ToPILImage('RGB')

    save_dirs = create_save_dir(args)

    iterations = int(args.max_nr / args.bs)

    for it, class_id in tqdm(enumerate(class_ids), total=iterations):
        
        if it >= args.max_nr:
            break

        output = pipe(class_labels=[class_id], num_inference_steps=25, generator=generator)

        pil_img = output.images[0] #trans_pil_img(output.images[0])

        if ".pt" in fileformat:
            tmp = np.array(img[0], dtype='float32')
            generated_images.append(tmp)

        if ".png" in fileformat:
            save_images(args, pil_img, it, [".png"], save_dirs)
        if ".jpg" in fileformat:
            save_images(args, pil_img, it, [".jpg"], save_dirs)

        if '.pt' in fileformat:      
            batch = torch.from_numpy(np.transpose(np.vstack(generated_images), (0,3,1,2)))
            save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
            save_to_pt(args, ws_gen_path, batch, args.save_gen_adv)


def calc_mean_std(image_loader):
    # https://iq.opengenus.org/calculate-mean-and-std-of-image-dataset/
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])

    for inputs, _ in tqdm(image_loader):
        total_sum += inputs.sum(axis = [0, 2, 3])
        total_sum_square += (inputs ** 2).sum(axis = [0, 2, 3])

    count = len(image_loader.dataset) * inputs.shape[-1] * inputs.shape[-1]

    # mean and std
    total_mean = total_sum / count
    total_var  = (total_sum_square / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    return total_mean, total_std


def calc_mean_std_without_target(image_loader):
    # https://iq.opengenus.org/calculate-mean-and-std-of-image-dataset/
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])

    for inputs in tqdm(image_loader):
        inputs = inputs['images']
        total_sum += inputs.sum(axis = [0, 2, 3])
        total_sum_square += (inputs ** 2).sum(axis = [0, 2, 3])

    count = len(image_loader.dataset) * inputs.shape[-1] * inputs.shape[-1]

    # mean and std
    total_mean = total_sum / count
    total_var  = (total_sum_square / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    return total_mean, total_std


def calc_mean_std_without_target2(image_loader):
    # https://iq.opengenus.org/calculate-mean-and-std-of-image-dataset
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])

    for inputs in image_loader:
        breakpoint()
        # inputs = inputs['images']
        total_sum += inputs.sum(axis = [0, 2, 3])
        total_sum_square += (inputs ** 2).sum(axis = [0, 2, 3])

    count = len(image_loader.dataset) * inputs.shape[-1] * inputs.shape[-1]

    # mean and std
    total_mean = total_sum / count
    total_var  = (total_sum_square / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    return total_mean, total_std


def calc_mean_std_without_target3(image_loader):
    # https://iq.opengenus.org/calculate-mean-and-std-of-image-dataset
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])

    for inputs in image_loader:
        total_sum += inputs.sum(axis = [0, 2, 3])
        total_sum_square += (inputs ** 2).sum(axis = [0, 2, 3])

    count = len(image_loader.dataset) * inputs.shape[-1] * inputs.shape[-1]

    # mean and std
    total_mean = total_sum / count
    total_var  = (total_sum_square / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    return total_mean, total_std