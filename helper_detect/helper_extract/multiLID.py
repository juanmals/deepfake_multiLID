import numpy as np
import os, sys
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from helper_extract.registrate_featuremaps import (
    registrate_featuremaps,
    get_feature_extractor,
)

from helper_extract.utils import (
    change_dict_device,
    feature_dict_to_list_timm
)


@torch.no_grad()
def multiLID_timm(args, images, images_advs, model):
    # get references to wanted activation layers of different networks
    # the number of activation layers is the number of featues for the LID
    act_layers = args.layers
    k, batch_size = 10, 100 # get_k(args)

    model.cpu()

    if args.extract in ['wb-multiLID', 'LID']:
        feature_extractor = get_feature_extractor(model, args.layers)
    
    def mle_batch(data, batch, k):
        data  = np.asarray(data,  dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)

        # select number of neighbors
        k = min(k, len(data)-1)
        
        ##########################################################################
        if args.extract == 'LID':
            f2  = lambda v: - k / np.sum(np.log(v/v[-1], where=v/v[-1]>0), dtype=np.float32) # original LID
        else:
            f2 = lambda v: - np.log(v/v[-1], where=v/v[-1]>0,  dtype=np.float32) # multiLID
        
        ##########################################################################
        dist = cdist(batch, data)
        dist = np.apply_along_axis(np.sort, axis=1, arr=dist)[:,1:k+1]
        multi_lid = np.apply_along_axis(f2, axis=1, arr=dist)

        return multi_lid

    lid_dim = len(act_layers) # number of featueres
    shape = np.shape(images[0])
    
    def estimate(i_batch):
        # estimation of the MLE batch
        start = i_batch * batch_size
        end = np.minimum(len(images), (i_batch + 1) * batch_size)
        n_feed = end - start
        
        # prepare data structure 
        if args.extract == 'LID':
            lid_batch     = np.zeros(shape=(n_feed, lid_dim))
            lid_batch_adv = np.zeros(shape=(n_feed, lid_dim))
        else:
            lid_batch     = np.zeros(shape=(n_feed, k, lid_dim))
            lid_batch_adv = np.zeros(shape=(n_feed, k, lid_dim))
        
        batch     = torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        batch_adv = torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        
        for j in range(n_feed):
            batch[j,:,:,:]     = images[j]
            batch_adv[j,:,:,:] = images_advs[j]
        
        range_lid_dim = lid_dim
        # extract feature maps from selected ReLU layers
        if args.extract in ['wb-multiLID', 'LID']:
            X_act = feature_extractor(batch)
            X_act     = feature_dict_to_list_timm(args, X_act)
            X_adv_act = feature_extractor(batch_adv)
            X_adv_act = feature_dict_to_list_timm(args, X_adv_act)

            # X_act = feature_extractor(batch.cuda())
            # X_act     = feature_dict_to_list_timm(args, X_act)
            # X_adv_act = feature_extractor(batch_adv.cuda())
            # X_adv_act = feature_dict_to_list_timm(args, X_adv_act)

        elif args.extract == 'bb-multiLID':
                X_act = batch
                X_adv_act = batch_adv
                range_lid_dim = lid_dim-1
            
        for i in range(range_lid_dim):
            X_act[i]      = np.asarray( X_act[i].cpu().detach().numpy()    , dtype=np.float32).reshape((n_feed, -1) )
            X_adv_act[i]  = np.asarray( X_adv_act[i].cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1) )
            
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            tmp_batch     = mle_batch( X_act[i], X_act[i]    , k=k )
            tmp_batch_adv = mle_batch( X_act[i], X_adv_act[i], k=k )   
            
            if args.extract == 'LID':
                lid_batch[:, i]       = tmp_batch
                lid_batch_adv[:, i]   = tmp_batch_adv
            else:
                lid_batch[:, :, i]       = tmp_batch
                lid_batch_adv[:, :, i]   = tmp_batch_adv

            
        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    
    n_batches = int(np.ceil(len(images) / float(batch_size)))
    
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
    
    characteristics         = np.asarray(lids, dtype=np.float32)
    characteristics_adv     = np.asarray(lids_adv, dtype=np.float32)

    # charactersitics...     of multiLID of clean data
    # charactersitics_adv... of multiLID of adv counterpart
    return characteristics, characteristics_adv
