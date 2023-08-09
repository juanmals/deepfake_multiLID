import os
import pdb
from tqdm import tqdm
import shutil
import numpy as np
import torch
import torchvision
from torchvision import transforms
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline
from datasets import load_dataset
import argparse

from cfg import *

from misc import (
    args_handling,
    print_args,
    create_dir,
    save_to_pt   
)

from helper_gen.helper_gen_images import (
    generate_images,
    generate_prompt_images,
)

from helper_gen.celebahqdataset import (
    get_celebaHQ_trainingloader
)

if __name__ == "__main__":

    # processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=[
                                                                'cifar10', 
                                                                'butterflies32', 
                                                                'celebaHQ256', 
                                                                'celebaHQ', 
                                                                'oxfordflowers64',
                                                                '...'
                                                                ],
                                                                help="")

    parser.add_argument("--mode", default="nor", choices=['nor', 'adv'], help="nor, adv")
    parser.add_argument("--bs", default=2000, help="batch size")

    parser.add_argument("--save_gen_nor",  default="nor_train.pt", help="save_gen_nor")
    parser.add_argument("--save_gen_adv",  default="", help="save_gen_adv")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()

    args = args_handling(args, parser, cfg_gen_path)
    print_args(args)

    if args.dataset == 'cifar10':
        if args.mode == 'nor':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_path, args.dataset), train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=4)

            batch = torch.Tensor([])
            for i, (img, _) in enumerate(tqdm(trainloader)):
                batch = img
                break

            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            model_id = "google/ddpm-cifar10-32"
            if "ddpm" in args.save_gen_adv:
                if "_ema" in args.save_gen_adv:
                    raise NotImplementedError("Install and execute: https://github.com/pesser/pytorch_diffusion ; python pytorch_diffusion/diffusion.py ema_cifar10 100 20 ; cp -r /home/lorenzp/DeepfakeDetection/src/pytorch-diffusion/results/ema_cifar10 /home/lorenzp/workspace/DeepfakeDetection/results/gen/cifar10")
                else:
                    pipeline = DDPMPipeline.from_pretrained(model_id).to("cuda") 
            elif "ddim" in args.save_gen_adv:
                pipeline = DDIMPipeline.from_pretrained(model_id).to("cuda") 
            elif "pndm" in args.save_gen_adv:
                pipeline = PNDMPipeline.from_pretrained(model_id).to("cuda")
            else:
                raise NotImplementedError 

            generate_images(args, pipeline)


    elif args.dataset == 'mnist':
        if args.mode == 'nor':
            from helper_gen.mnist import ( train_loader )
            train_dataloader = train_loader(args, image_size=28)
            batch = next(iter(train_dataloader))["images"]
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            # model_id = "dimpo/ddpm-mnist"
            model_id = "nabdan/mnist"
            if "ddpm" in args.save_gen_adv:
                pipeline = DDPMPipeline.from_pretrained(model_id).to("cuda")
            else:
                raise NotImplementedError 
            generate_images(args, pipeline)


    elif args.dataset == 'butterflies32':
        if args.mode == 'nor':
            from helper_gen.butterflies import ( train_loader )

            train_dataloader = train_loader(args, image_size=32)

            batch = next(iter(train_dataloader))["images"]
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            pipeline = DDPMPipeline.from_pretrained('clp/sd-class-butterflies-32').to("cuda")
            generate_images(args, pipeline)


    elif args.dataset == 'pokemon64':
        if args.mode == 'nor':
            from helper_gen.oxfordflowers import ( train_loader )
            train_dataloader = train_loader(args, image_size=64)
            batch = next(iter(train_dataloader))["images"]
            # batch = generate_images(args, pipeline)
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)
        elif args.mode == 'adv':
            if "ddpm" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained('anton-l/ddpm-ema-pokemon-64').to("cuda") # https://huggingface.co/anton-l/ddpm-ema-pokemon-64
            else:
                raise NotImplementedError 
            generate_images(args, pipeline)


    elif args.dataset == 'oxfordflowers64':
        if args.mode == 'nor':
            from helper_gen.oxfordflowers import ( train_loader )
            train_dataloader = train_loader(args, image_size=64)
            batch = next(iter(train_dataloader))["images"]
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)
        elif args.mode == 'adv':
            if "ddpm" in args.save_gen_adv:
                if "_ema" in args.save_gen_adv:
                    pipeline = DDPMPipeline.from_pretrained('mrm8488/ddpm-ema-flower-64').to("cuda") # https://huggingface.co/shahp7575/oxford_flowers_diffused
                else:
                    raise NotImplementedError("No oxford flowers 64 ema found.")
            else:
                raise NotImplementedError 
            generate_images(args, pipeline)


    elif args.dataset == 'butterflies128':
        if args.mode == 'nor':
            from helper_gen.butterflies import ( train_loader )
            train_dataloader = train_loader(args, image_size=128)
            batch = next(iter(train_dataloader))["images"]
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            if "_ema" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("mrm8488/ddpm-ema-butterflies-128").to("cuda")
            else:
                pipeline = DDPMPipeline.from_pretrained('xud/ddpm-butterflies-128').to("cuda")
            
            generate_images(args, pipeline)


    elif args.dataset == 'celebaHQ128':
        if args.mode == 'nor':
            train_dataloader = get_celebaHQ_trainingloader(args, data='Hair_Color', img_size=128, batch_size=args.bs, num_workers=8, shuffle=True)
            batch = next(iter(train_dataloader))[0]
            print("shape> ", batch.shape)
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            
            # load model and scheduler
            if "ddpm" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("simlightvt/ddpm-celebahq-128").to("cuda") 
            else:
                raise NotImplementedError 

            generate_images(args, pipeline)


    elif args.dataset == 'celebaHQ256':
        if args.mode == 'nor':
            train_dataloader = get_celebaHQ_trainingloader(args, data='Hair_Color', img_size=256, batch_size=args.bs, num_workers=8, shuffle=True)
            batch = next(iter(train_dataloader))[0]
            print("shape> ", batch.shape)
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            model_id = "google/ddpm-celebahq-256"
            if "_ema" in args.save_gen_adv:
                model_id = "google/ddpm-ema-celebahq-256"

            # load model and scheduler
            if "ddpm" in args.save_gen_adv:
                pipeline = DDPMPipeline.from_pretrained(model_id).to("cuda")
            elif "ddim" in args.save_gen_adv:
                pipeline = DDIMPipeline.from_pretrained(model_id).to("cuda")
            elif "pndm" in args.save_gen_adv:
                pipeline = PNDMPipeline.from_pretrained(model_id).to("cuda")
            elif "ncsnpp" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("google/ncsnpp-celebahq-256").to("cuda") 
            elif "ldm" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256").to("cuda")
            else:
                raise NotImplementedError 
            
            generate_images(args, pipeline)


    elif args.dataset == 'lsun-bedroom256':
        if args.mode == 'nor':
            from helper_gen.lsun import ( train_loader )
            train_dataloader = train_loader(args, image_size=256)
            iterations = int(args.max_nr / args.bs)

            images = []
            for it, img in tqdm(enumerate(train_dataloader), total=iterations):
                tmp_img = img["images"]
                images.append(tmp_img)
                if it+1 == iterations:
                    break
            batch = torch.vstack(images)
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':

            if (
                "ddpm" in args.save_gen_adv or 
                "ddim" in args.save_gen_adv or 
                "pndm" in args.save_gen_adv or
                "ncsnpp" in args.save_gen_adv
                ):
                model_id = "google/ddpm-bedroom-256"
                if "_ema" in args.save_gen_adv:
                    model_id = "google/ddpm-ema-bedroom-256"

                if "ddpm" in args.save_gen_adv:
                    pipeline = DDPMPipeline.from_pretrained(model_id).to("cuda") # https://huggingface.co/fusing/ddpm-lsun-bedroom
                elif "ddim" in args.save_gen_adv:
                    pipeline = DDIMPipeline.from_pretrained(model_id).to("cuda")
                elif "pndm" in args.save_gen_adv:
                    pipeline = PNDMPipeline.from_pretrained(model_id).to("cuda")
                elif "ncsnpp" in args.save_gen_adv:
                    pipeline = DiffusionPipeline.from_pretrained("google/ncsnpp-bedroom-256").to("cuda")

                else:
                    raise NotImplementedError("not found!")
            
                generate_images(args, pipeline)

            elif "sdv21" in args.save_gen_adv:
                # https://huggingface.co/stabilityai/stable-diffusion-2-1
                model_id = "stabilityai/stable-diffusion-2-1"

                # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
                pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                pipeline = pipeline.to("cuda")

                generate_prompt_images(args, pipeline, prompt="a photo of a bedroom", resize=256)
            
            elif "vqd" in args.save_gen_adv:
                from diffusers import VQDiffusionPipeline
                pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16)
                pipeline = pipeline.to("cuda")
            
                generate_prompt_images(args, pipeline, prompt="a photo of a bedroom")

            elif "ldm" in args.save_gen_adv:
                model_id = "CompVis/ldm-text2im-large-256"
                pipeline = DiffusionPipeline.from_pretrained(model_id)

                kwargs_tmp = {"num_inference_steps": 50, "eta": 0.3, "guidance_scale": 6}
                pipeline = pipeline.to("cuda")

                generate_prompt_images(args, pipeline, prompt="a photo of a bedroom", resize=0, kwargs_tmp=kwargs_tmp)


    elif args.dataset == 'lsun-cat256':
        if args.mode == 'nor':
            from helper_gen.dataset_tools import ( open_dataset )
            import torchvision
            dir_folder = "/home/DATA/ITWM/lorenzp/lsun/cat_lmdb"
            nr, gen  = open_dataset(dir_folder, max_images=args.max_nr)

            images = []
            for i, img in enumerate(gen):
                print("i> ", i)
                transposed = img['img'].transpose((2, 0, 1)).copy()
                trans2 = torchvision.transforms.CenterCrop(256)(torch.from_numpy(transposed))
                # trans2 = torchvision.transforms.RandomCrop(256)(torch.from_numpy(transposed))
                #trans2 = torchvision.transforms.Resize((256,256))(torch.from_numpy(transposed))
                images.append( trans2 / 255.)

            batch = torch.stack(images)
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)


        elif args.mode == 'adv':
            model_id = "google/ddpm-cat-256"
            if "_ema" in args.save_gen_adv:
                model_id = "google/ddpm-ema-cat-256"

            if "ddpm" in args.save_gen_adv:
                pipeline = DDPMPipeline.from_pretrained(model_id).to("cuda")
            elif "ddim" in args.save_gen_adv:
                pipeline = DDIMPipeline.from_pretrained(model_id).to("cuda") 
            elif "pndm" in args.save_gen_adv:
                pipeline = PNDMPipeline.from_pretrained(model_id).to("cuda")
            else:
                raise NotImplementedError
            
            generate_images(args, pipeline)


    elif args.dataset == 'lsun-church256':
        if args.mode == 'nor':
            from helper_gen.dataset_tools import ( open_dataset )
            import torchvision
            dir_folder = "/home/DATA/ITWM/lorenzp/lsun/church/church_outdoor_train_lmdb/"
            nr, gen  = open_dataset(dir_folder, max_images=args.max_nr)

            images = []

            for i, img in enumerate(gen):
                print("i> ", i)
                transposed = img['img'].transpose((2, 0, 1)).copy()
                trans2 = torchvision.transforms.CenterCrop(256)(torch.from_numpy(transposed))
                # trans2 = torchvision.transforms.RandomCrop(256)(torch.from_numpy(transposed))
                # trans2 = torchvision.transforms.Resize((256,256))(torch.from_numpy(transposed))
                images.append( trans2 / 255.)

            batch = torch.stack(images)            
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            model_id = "google/ddpm-church-256"
            if "_ema" in args.save_gen_adv:
                model_id = "google/ddpm-ema-church-256"

            if "ddpm" in args.save_gen_adv:
                pipeline = DDPMPipeline.from_pretrained(model_id).to("cuda")
            elif "ddim" in args.save_gen_adv:
                pipeline = DDIMPipeline.from_pretrained(model_id).to("cuda") 
            elif "pndm" in args.save_gen_adv:
                pipeline = PNDMPipeline.from_pretrained(model_id).to("cuda")
            elif "ncsnpp" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("google/ncsnpp-church-256").to("cuda")
            else:
                raise NotImplementedError 
            generate_images(args, pipeline)


    elif args.dataset == 'ffhq256':
        if args.mode == 'nor':
            from helper_gen.ffhq import ( load_ffhq )
            batch = load_ffhq(size=args.max_nr, img_size=256)
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            if "ncsnpp" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("google/ncsnpp-ffhq-256").to("cuda")
            else:
                raise NotImplementedError 

            generate_images(args, pipeline)


    elif args.dataset == 'ffhq1024':
        if args.mode == 'nor':
            from helper_gen.ffhq import ( load_ffhq )
            batch = load_ffhq(size=args.max_nr)
            save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

        elif args.mode == 'adv':
            if "ncsnpp" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("google/ncsnpp-ffhq-1024").to("cuda")
            else:
                raise NotImplementedError
            generate_images(args, pipeline)


    elif args.dataset == 'celebaHQ':
        if args.mode == 'nor':
            pass
        elif args.mode == 'adv':
            if "pggan" in args.save_gen_adv:
                pipeline = DiffusionPipeline.from_pretrained("huggan/pggan-celebahq-1024").to("cuda") # https://huggingface.co/huggan/pggan-celebahq-1024/blob/main/README.md https://huggingface.co/fusing/ddpm-celeba-hq
            else:
                raise NotImplementedError

            generate_images(args, pipeline)


    elif args.dataset == 'cifake':
        from helper_gen.cifake import ( generate_training_samples )
        batch = generate_training_samples(args)
        save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

    # elif args.dataset == 'af-church':
    #     from helper_gen.artifact import ( generate_training_samples )
    #     batch = generate_training_samples(args, 'church')
    #     save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

    # elif args.dataset == 'af-bedroom':
    #     from helper_gen.artifact import ( generate_training_samples )
    #     batch = generate_training_samples(args, 'bedroom')
    #     save_to_pt(args, ws_gen_path, batch, args.save_gen_nor)

    elif args.dataset == 'ddpm_af-church':
        from helper_gen.artifact import ( generate_training_samples )
        batch = generate_training_samples(args, 'church')

        save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
        save_to_pt(args, ws_gen_path, batch, save_path)


    elif args.dataset == 'ddpm_af-bedroom':
        from helper_gen.artifact import ( generate_training_samples )
        batch = generate_training_samples(args, 'bedroom')

        save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
        save_to_pt(args, ws_gen_path, batch, save_path)


    elif args.dataset == 'af-all':
        from helper_gen.artifact import ( generate_training_samples_all )
        batch = generate_training_samples_all(args)

        save_path = args.save_gen_nor if args.mode == 'nor' else args.save_gen_adv
        save_to_pt(args, ws_gen_path, batch, save_path)


    elif args.dataset == 'stablediffv21_laion768':
        image_size = 768
        if args.mode == 'nor':
            print("img2dataset")
            from helper_gen.laion import ( load_hf_dataset, save_kaionHQ_dataset )
            dataset_id = "laion/laion-high-resolution" 
            dataset = load_hf_dataset(args, dataset_id)

            # save_kaionHQ_dataset(args, dataset, image_size)


        elif args.mode == 'adv':
            # https://huggingface.co/stabilityai/stable-diffusion-2-1
            model_id = "stabilityai/stable-diffusion-2-1"

            # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to("cuda")

            generate_prompt_images(args, pipeline)

    elif args.dataset == 'dalle2':
        if args.mode == 'nor':
            print("img2dataset")
            # from helper_gen.laion import ( load_hf_dataset, save_kaionHQ_dataset )
            # dataset_id = "laion/laion-high-resolution" 
            # dataset = load_hf_dataset(args, dataset_id)

            # save_kaionHQ_dataset(args, dataset, image_size)


        elif args.mode == 'adv':
            # https://huggingface.co/stabilityai/stable-diffusion-2-1
            model_id = "stabilityai/stable-diffusion-2-1"

            # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to("cuda")

            generate_prompt_images(args, pipeline)


    elif args.dataset == 'midjourney':
        # https://github.com/lucidrains/imagen-pytorch
        if args.mode == 'nor':
            print("img2dataset")
            # from helper_gen.laion import ( load_hf_dataset, save_kaionHQ_dataset )
            # dataset_id = "laion/laion-high-resolution" 
            # dataset = load_hf_dataset(args, dataset_id)

            # save_kaionHQ_dataset(args, dataset, image_size)

        elif args.mode == 'adv':
            # https://huggingface.co/stabilityai/stable-diffusion-2-1
            model_id = "stabilityai/stable-diffusion-2-1"

            # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to("cuda")

            generate_prompt_images(args, pipeline)

    elif args.dataset == 'imagen':
        # https://github.com/lucidrains/imagen-pytorch
        print("Not implemented, because no pretrained weights available!")

    elif args.dataset == 'diffusiondb512':
        if args.mode == 'nor':
            print("img2dataset")

            if "nor_train_crop_jpg" == args.save_gen_nor:
                print(
                    # mkdir /home/lorenzp/workspace/DeepfakeDetection/results/gen/diffusiondb
                    # cd /home/lorenzp/workspace/DeepfakeDetection/results/gen/diffusiondb
                    # mkdir nor_train_crop_jpg

                    # conda activate advtrain_julia

                    # img2dataset --url_list /home/lorenzp/DeepfakeDetection/analysis/stablediffv21/hq15k.parquet --input_format "parquet"    --url_col "URL" --caption_col "TEXT" --output_format files       --output_folder laion-high-resolution-output-512-crop_15k --processes_count 16 --thread_count 64 --image_size 512       --resize_only_if_bigger=False --resize_mode="center_crop" --skip_reencode=True        --save_additional_columns '["similarity","hash","punsafe","pwatermark","LANGUAGE"]' --enable_wandb True

                    # rm -rf nor_train_crop_jpg/*.jpg
                    # cp -r laion-high-resolution-output-512-crop_15k/00000/*.jpg  nor_train_crop_jpg/
                    # cp -r laion-high-resolution-output-512-crop_10k/00000/*.jpg  nor_train_crop_jpg/
                    # ls -l nor_train_crop_jpg/  | grep -v ^l | wc -l
                    # rm -rf crop.txt
                    # ls nor_train_crop_jpg/*.jpg > crop.txt
                    # ls adv_large_random_5k_jpg/*.jpg > adv_large_random_5k_jpg.txt
                )

            elif "laion2B_en_aesthetic" in args.save_gen_nor:

                from helper_gen.laion import ( load_hf_dataset )
                dataset_id = "laion/laion2B-en-aesthetic"
                dataset = load_hf_dataset(args, dataset_id)

            elif 'sac512' in  args.save_gen_nor:
                from helper_gen.sac import (  get_sac_trainingloader )
                from helper_gen.helper_gen_images import ( generate_images_from_hf )

                train_loader = get_sac_trainingloader(args, data='images', batch_size=args.bs, num_workers=0)
                generate_images_from_hf(args, train_loader)


        elif args.mode == 'adv':
            from helper_gen.diffusiondb import (  train_loader )

            if "." in  args.save_gen_adv:
                model_id = args.save_gen_adv.split(".")[0]
            elif "_jpg" in  args.save_gen_adv:
                model_id = args.save_gen_adv.split("_jpg")[0]

            print("model_id> ", model_id)

            train_loader = train_loader(args, model_id, image_size=512)
            from helper_gen.helper_gen_images import ( generate_images_from_hf )
            generate_images_from_hf(args, train_loader)


    elif args.dataset == 'sac512':
        if args.mode == 'nor':
            pass

        elif args.mode == 'adv':
            from helper_gen.sac import (  get_sac_trainingloader )
            from helper_gen.helper_gen_images import ( generate_images_from_hf )

            train_loader = get_sac_trainingloader(args, data='images', batch_size=args.bs, num_workers=0)
            generate_images_from_hf(args, train_loader)


    elif args.dataset == 'imagenet':
        if args.mode == 'nor':
            from helper_gen.imagenet import( loat_dataset )
            from helper_gen.helper_gen_images import ( generate_images_from_loader )
            data_loader = loat_dataset(args)

            generate_images_from_loader(args, data_loader)

        elif args.mode == 'adv':
            classes = "/home/DATA/ITWM/lorenzp/ImageNet/imagenet_classes.txt"

            if "dit" in args.save_gen_adv:
                from diffusers import DiTPipeline, DPMSolverMultistepScheduler
                import random
                from helper_gen.helper_gen_images import ( generate_images_from_dit )

                pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe = pipe.to("cuda")

                # pick words from Imagenet class labels
                available_words = pipe.labels 
    
                class_id = list(available_words.values())
                tmp_nr = args.max_nr  - len(class_id)

                tmp_add = random.sample(class_id,tmp_nr)
                class_ids = class_id + tmp_add

                generator = torch.manual_seed(33)
                generate_images_from_dit(args, class_ids, pipe, generator)


            elif "mdt" in args.save_gen_adv:
                from huggingface_hub import snapshot_download

                models_path = snapshot_download("shgao/MDT-XL2", cache_dir="/home/scratch/adversarialml/mkd")
                breakpoint()
                ckpt_model_path = os.path.join(models_path, "mdt_xl2_v1_ckpt.pt")
                
                

    else: 
        raise NotImplementedError("Err: Dataset not found!", args.dataset)


    # print("shape: ", batch.shape)