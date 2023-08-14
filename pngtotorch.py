import os, pdb
import numpy as np
import matplotlib.image as mpimg
import torch
import natsort 


def save_to_png(original_data, pth):
    for i, data in enumerate(original_data):
        mpimg.imsave(os.path.join(pth, str(i) + '.png'), data)

def get_image_names(pth):
    myimages = [] #list of image filenames
    dirFiles = os.listdir(pth) #list of directory files
    dirFiles = natsort.natsorted(dirFiles,reverse=False)

    for files in dirFiles: # filter out all non jpgs
        if '.png' in files:
            myimages.append(files)
            
    return myimages
        
def load_png(pth):
    image_names = get_image_names(pth)
    
    loaded_images = []
    for image_name in image_names:
        cur_img = mpimg.imread( os.path.join(pth, image_name) )[:,:,:3].transpose((2,0,1))
        loaded_images.append(cur_img)
        
    stacked_images = torch.from_numpy(np.stack(loaded_images, axis=0))
    
    return stacked_images


if __name__ == "__main__":
    print("start!")

    nor_pth = "/home/lorenzp/workspace/DeepfakeDetection/results/gen/butterfly32/nor_train_png.pt"
    adv_pth = "/home/lorenzp/workspace/DeepfakeDetection/results/gen/butterfly32/adv_ddpm_png.pt"
    
    nor_png_pth = "/home/lorenzp/workspace/DeepfakeDetection/results/gen/png_butterfly32/nor"
    adv_png_pth = "/home/lorenzp/workspace/DeepfakeDetection/results/gen/png_butterfly32/adv"
    
    output_pth = "/home/lorenzp/workspace/DeepfakeDetection/results/gen/butterfly32"
    
    nor_img = load_png(nor_png_pth)         
    # torch.save(nor_img, os.path.join(os.path.dirname(nor_png_pth), 'nor_train.pt'))
    torch.save(nor_img, os.path.join(output_pth, 'nor_train_png.pt'))
    
    adv_img = load_png(adv_png_pth)
    # torch.save(adv_img, os.path.join(os.path.dirname(adv_png_pth), 'adv_ddpm.pt'))
    torch.save(adv_img, os.path.join(output_pth, 'adv_ddpm_png.pt'))
    
    # import pdb; pdb.set_trace()

    print("finished!")