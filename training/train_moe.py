

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from torch.utils.data import ConcatDataset
from detectors.fake_segment_MSE_MOE import Moe
from detectors.k_organ_MOE1 import CLIPVisionTransformer_K

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='training/config/detector/clip_fine_MSE_moe_k.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
        #train_set = VLFFDRegionDataset(config=config,mode='train',)
        train_set = CombinedVLFFDSBIDataset(config=config, mode='train')
        # train_set2 = SBIDataset(config, mode='train')
        # train_set = ConcatDataset([train_set1, train_set2])

    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
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
                drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring
    
def set_moe(model):
    ckpt_pth = '/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/logs/training/clip_fine_text_MSE_2025-12-08-00-04-07/test/avg/ckpt_best.pth'
    ckpt = torch.load(ckpt_pth, map_location=device)
    model.load_state_dict(ckpt,strict=True)
    print('model load successfully')
    layers = range(5, 12)
    for layer_indice in layers:
        mlp = model.model.vision_model.encoder.layers[layer_indice].mlp
        new_mlp = Moe()
        for expert in new_mlp.experts:
            expert[0].weight.data.copy_(mlp.fc1.weight.data)
            expert[0].bias.data.copy_(mlp.fc1.bias.data)
            expert[2].weight.data.copy_(mlp.fc2.weight.data)
            expert[2].bias.data.copy_(mlp.fc2.bias.data)
        model.model.vision_model.encoder.layers[layer_indice].mlp = new_mlp
    return model
    
def set_k_moe(model):
    ckpt_pth = '/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/logs/training/clip_fine_text_MSE_2025-12-08-00-04-07/test/avg/ckpt_best.pth'  #
    # ckpt_pth = '/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/logs/training/clip_fine_text_MSE_2026-01-26-19-29-39/test/avg/ckpt_best.pth'  #
    ckpt = torch.load(ckpt_pth, map_location=device)
    model.load_state_dict(ckpt,strict=True)
    original_vision_state_dict = model.model.vision_model.state_dict()
    print('pretrained model load successfully!!!')
    vision_cfg = model.model.vision_model.config
    model.model.vision_model = CLIPVisionTransformer_K(vision_cfg).to(device)
    #print(original_vision_state_dict.keys())
    #print(model.model.vision_model.state_dict().keys())
    new_ckpt = original_vision_state_dict
    for i in range(0,12,6):
        for j in range(5):
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.weight'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.weight']
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.bias'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.bias']
    #print(original_vision_state_dict['encoder.layers.3.self_attn.k_proj.bias'])
    #print(new_ckpt['encoder.layers.3.self_attn.attention_k.k_moe.0.bias'])
    #print(new_ckpt.keys())
    model.model.vision_model.load_state_dict(new_ckpt, True)
    #print(list(model.named_parameters())[0:5])
    #input()
    print('vision model changed and load successfully!!!')
    return model
    
def set_retrain_k_moe(model):
    original_vision_state_dict = model.model.vision_model.state_dict()  # 原始clip vit参数
    vision_cfg = model.model.vision_model.config
    model.model.vision_model = CLIPVisionTransformer_K(vision_cfg).to(device)
    new_ckpt = original_vision_state_dict
    for i in range(0,12,6):
        for j in range(5):
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.weight'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.weight']
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.bias'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.bias']
    model.model.vision_model.load_state_dict(new_ckpt, strict=True)
    print('vision model has been exchanged!!!')
    return model
    
def set_train_only_k_moe(model):
    ckpt_pth = ''    # put your pre-trained first stage weight here
    ckpt = torch.load(ckpt_pth, map_location=device)
    model.load_state_dict(ckpt,strict=True)
    original_vision_state_dict = model.model.vision_model.state_dict()
    print('pretrained model load successfully!!!')
    vision_cfg = model.model.vision_model.config
    model.model.vision_model = CLIPVisionTransformer_K(vision_cfg).to(device)
    new_ckpt = original_vision_state_dict
    for i in range(0,12,2):
        for j in range(5):
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.weight'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.weight']
            new_ckpt[f'encoder.layers.{i}.self_attn.attention_k.k_moe.{j}.bias'] = new_ckpt[f'encoder.layers.{i}.self_attn.k_proj.bias']
    model.model.vision_model.load_state_dict(new_ckpt,strict=True)
    print('vision model changed and load successfully!!!')
    for param in model.parameters():
        param.requires_grad = False
    train_param = 'attention_k.k_moe'
    for name, param in model.named_parameters():
        if train_param in name:
            param.requires_grad = True
    print('only train k parameters!!!')
    return model

    

def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json'
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name']  + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    if config['model_name'] == 'clip_fine_text_MSE_moe':
        model = set_moe(model)
    elif config['model_name'] == 'clip_fine_text_MSE_moe_k':
        #model = set_k_moe(model)
        #model = set_retrain_k_moe(model)
        model = set_train_only_k_moe(model)
    

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    main()
