#!/usr/bin/env python3

import os, sys, pdb
import torch 
import numpy as np
from tqdm import tqdm
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


def test_activation(args, logger, model, images):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 5
    model.conv3[1].residual[1].register_forward_hook(get_activation('conv3_1_relu_1'))
    layers = [
        'conv3_1_relu_1'
    ]

    print(activation)
    pdb.set_trace()

    model(images[0])
    print(activation)

    return activation

@torch.no_grad()
def get_feature_extractor(model, layers):
    return create_feature_extractor(model, return_nodes=layers)


@torch.no_grad()
def extract_features_timm(args, loader, model):
    feature_extractor = get_feature_extractor(model.cpu(), args.layers)

    for img in tqdm(loader):
        model_feature_maps = feature_extractor(img.cpu())
       
    return model_feature_maps



@torch.no_grad()
def registrate_featuremaps(args, model):
    layers = args.layers
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu()
        return hook

    def get_layer_feature_maps(activation_dict, act_layer_list):
        act_val_list = []
        for it in act_layer_list:
            act_val = activation_dict[it]
            act_val_list.append(act_val.cpu())
        return act_val_list

    if args.model == 'rn18':
        nr_layers = len(layers)
        h1 = model.layer4[0].conv2.register_forward_hook( get_activation(layers[0]) ) # last layer
        h2 = model.layer4[1].conv2.register_forward_hook( get_activation(layers[1]) ) # previous
        if nr_layers > 2:
            h3 = model.layer3[0].conv2.register_forward_hook( get_activation(layers[2]) )
            h4 = model.layer3[1].conv2.register_forward_hook( get_activation(layers[3]) )
        if nr_layers > 4:
            h5 = model.layer2[0].conv2.register_forward_hook( get_activation(layers[4]) )
            h6 = model.layer2[1].conv2.register_forward_hook( get_activation(layers[5]) )
        if nr_layers > 6:
            h7 = model.layer1[0].conv2.register_forward_hook( get_activation(layers[6]) )
            h8 = model.layer1[1].conv2.register_forward_hook( get_activation(layers[7]) )
    else:
        raise NotImplementedError("Model not found!")
    
    return get_layer_feature_maps, layers, model, activation


@torch.no_grad()
def extract_features(args, loader, model):

    get_layer_feature_maps, layers, model, activation = registrate_featuremaps(args, model)

    for img in tqdm(loader):
        model( img.cuda() )

    return get_layer_feature_maps, layers, model, activation

