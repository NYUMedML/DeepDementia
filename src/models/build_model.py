import json
import pickle
import timeit
import sys
import numpy as np
import torch

from .classifier import LinearClassifierAlexNet
from .models import NetWork, SFCN, NetWork_contrastive
from .ResNet import ResNet18
from .LinearModel import alex_net_complete
from .EfficientNet import EfficientNet

def prepare_model(arch = 'ours', in_channel=1, 
    feat_dim=128, n_hid_main=200, n_label=3, 
    expansion = 8, type_name='conv3x3x3', norm_type = 'Instance'):

    if arch == 'ours':
        image_embeding_model = NetWork(in_channel=in_channel,feat_dim=feat_dim, expansion = expansion, type_name=type_name, norm_type=norm_type)
        classifier = LinearClassifierAlexNet(in_dim=feat_dim, n_hid=n_hid_main, n_label=n_label)
        main_model = alex_net_complete(image_embeding_model, classifier)
    elif arch == 'resnet':
        main_model = ResNet18(num_classes=n_label)
    elif arch == 'efficientnet':
        main_model = EfficientNet(arch_type = "efficientnet-b0", num_classes=n_label)
    elif arch == 'SFCN':
        main_model = SFCN(n_label, feat_dim)
    elif arch == 'contrastive':
        main_model = NetWork_contrastive(in_channel=in_channel,feat_dim=feat_dim, num_classes=n_label, expansion = expansion, type_name=type_name, norm_type=norm_type)
    # generate the classifier
    return main_model

def build_model(config, input_dim = 3):
    arch = config['model']['arch']
    in_channel = config['model']['input_channel']
    feat_dim = config['model']['feature_dim']
    n_label = config['model']['n_label']
    n_hid_main = config['model']['nhid']
    expansion = config['model']['expansion']
    type_name = config['model']['type_name']
    norm_type = config['model']['norm_type']

    main_model = prepare_model(arch = arch, in_channel=in_channel, feat_dim=feat_dim, 
        n_hid_main=n_hid_main,  n_label=n_label, 
        expansion= expansion, type_name=type_name, norm_type=norm_type)

    if config['training_parameters']['pretrain'] is not None:
        best_model_dir = './saved_model/'
        pretrained_dict = torch.load(best_model_dir+ config['training_parameters']['pretrain'] + '_model_low_loss.pth.tar',map_location='cpu')['state_dict']
        model_dict = main_model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:]in model_dict.keys())}
        model_dict.update(pretrained_dict) 
        print(model_dict.keys())
        main_model.load_state_dict(model_dict)
    return main_model