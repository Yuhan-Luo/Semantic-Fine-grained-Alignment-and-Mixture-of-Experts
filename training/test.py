"""
eval pretained model.
"""
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from sklearn import metrics

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detectors.fake_segment_MSE_MOE import Moe
from detectors.k_organ_MOE1 import CLIPVisionTransformer_K
import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='training/config/detector/clip_fine_MSE_moe_k.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")

parser.add_argument('--weights_path', type=str, 
                    default='/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/training/weights/best.pth')  # 
#parser.add_argument("--lmdb", action='store_true', default=False)
parser.add_argument('--cross_auc', type=bool, default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    #########
    pre_logits_lists = []

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        # feature_lists += list(predictions['feat'].cpu().detach().numpy())
        # pre_logits_lists += list(predictions['cls'][:,1].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists)  # ,np.array(feature_lists) , np.array(pre_logits_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}
    real_sample_all_datasets = {}
    fake_sample_all_datasets = {}
    real_pre_all_datasets = {}
    fake_pre_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps = test_one_dataset(model, test_data_loaders[key])
        # prediction_nps假概率值  logits_nps假文本得分

        label_true_indices = np.where(label_nps == 0)[0]
        label_fake_indices = np.where(label_nps == 1)[0]
        # draw_prob(predictions_nps[label_fake_indices], key)
        real_pre_all_datasets[key] = predictions_nps[label_true_indices]
        fake_pre_all_datasets[key] = predictions_nps[label_fake_indices]
        real_sample_all_datasets[key] = label_nps[label_true_indices]
        fake_sample_all_datasets[key] = label_nps[label_fake_indices]
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")
            if k == 'pred':
                tqdm.write(f"{len(v)}")

    if args.cross_auc:
        calculate_cross_auc(real_sample_all_datasets, fake_sample_all_datasets,real_pre_all_datasets,fake_pre_all_datasets,keys)

    return metrics_all_datasets

def calculate_cross_auc(real_sample, fake_sample,real_pre,fake_pre, keys):
    keys = list(keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            y_true = np.concatenate([fake_sample[keys[i]], real_sample[keys[j]]])
            y_pred = np.concatenate([fake_pre[keys[i]], real_pre[keys[j]]])
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            tqdm.write(f'cross-auc: {keys[i]} and {keys[j]}: {auc}')

def draw_prob(predictions, key):
    prob = []
    for i in range(0, len(predictions), 100):
        if predictions[i] >= 0.5:
            prob.append(predictions[i])
    print(prob)

def set_k_moe(model):
    ckpt_pth = '/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/logs/training/clip_fine_text_MSE_2025-12-08-00-04-07/test/avg/ckpt_best.pth'
    ckpt = torch.load(ckpt_pth, map_location=device)
    model.load_state_dict(ckpt,strict=True)
    original_vision_state_dict = model.model.vision_model.state_dict()
    print('pretrained model load successfully!!!')
    vision_cfg = model.model.vision_model.config
    model.model.vision_model = CLIPVisionTransformer_K(vision_cfg).to(device)
    #print(original_vision_state_dict.keys())
    #print(model.model.vision_model.state_dict().keys())
    new_ckpt = original_vision_state_dict
    for i in range(12):
        for j in range(5):
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.weight'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.weight']
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.bias'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.bias']
    #print(original_vision_state_dict['encoder.layers.3.self_attn.k_proj.bias'])
    #print(new_ckpt['encoder.layers.3.self_attn.attention_k.k_moe.0.bias'])
    #print(new_ckpt.keys())
    model.model.vision_model.load_state_dict(new_ckpt, True)
    print('vision model changed and load successfully!!!')
    return model
    
def load_k_moe(model):
    vision_cfg = model.model.vision_model.config
    model.model.vision_model = CLIPVisionTransformer_K(vision_cfg).to(device)

    ckpt_pth = ''  # put the model weight that you want to test here
    ckpt = torch.load(ckpt_pth, map_location=device)
    model.load_state_dict(ckpt, True)
    print('model changed and load successfully!!!')
    return model


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    if config['model_name'] == 'clip_fine_text_MSE_moe_k':
        #model = set_k_moe(model)
        model = load_k_moe(model)
    # print(model)
    #epoch = 0
    elif weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        # print(ckpt)
        # input()
        if config['model_name'] == 'effort':
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            new_weights = {}
            for key, value in ckpt.items():
                #print(key)
                new_key = key.replace('module.','')
                new_weights[new_key] = value
            new_weights['backbone.embeddings.position_ids'] = model.backbone.embeddings.position_ids
            # new_weights['backbone.embeddings.position_ids'] = torch.arange(257).unsqueeze(0)
        model.load_state_dict(ckpt, strict=True)
        #print(model)
        #input()
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
