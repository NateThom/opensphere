import os
import os.path as osp
import yaml
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabulate import tabulate
from datetime import datetime

from utils import fill_config
from builder import build_dataloader, build_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--config', 
            help='config files for testing datasets')
    parser.add_argument('--proj_dirs', '--list', nargs='+',
            help='the project directories to be tested')
    parser.add_argument('--start_time', 
            help='time to start training')
    args = parser.parse_args()

    return args

@torch.no_grad()
def get_feats(net, data, flip=True):
    # extract features from the original 
    # and horizontally flipped data
    feats = net(data)
    if flip:
        data = torch.flip(data, [3])
        feats += net(data)

    return feats.data.cpu()

@torch.no_grad()
def test_run(net, checkpoints, dataloaders):
    features = {}
    predictions = {}
    for n_ckpt, checkpoint in enumerate(checkpoints):
        # load model parameters
        net.load_state_dict(torch.load(checkpoint))
        for n_loader, dataloader in enumerate(dataloaders):
            # get feats from test_loader
            dataset_feats = []
            dataset_indices = []
            for n_batch, (data, indices) in enumerate(dataloader):
                # collect feature and indices
                data = data.cuda()
                indices = indices.tolist()
                feats = get_feats(net, data)
                dataset_feats.append(feats)
                dataset_indices.extend(indices)
                # progress
                print('feature extraction:',
                    'checkpoint: {}/{}'.format(n_ckpt+1, len(checkpoints)),
                    'dataset: {}/{}'.format(n_loader+1, len(dataloaders)),
                    'batch: {}/{}'.format(n_batch+1, len(dataloader)),
                    end='\r')
            print('')
            # eval
            dataset_feats = torch.cat(dataset_feats, dim=0)
            dataset_feats_indices = dataset_feats[dataset_indices]
            feats = F.normalize(feats, dim=1)
            scores = dataloader.dataset.get_scores(dataset_feats_indices)
            # save
            name = dataloader.dataset.name
            if name not in features:
                features[name] = dataset_feats
                predictions[name] = scores
            else:
                features[name] = torch.cat(
                    [features[name].unsqueeze(0), dataset_feats.unsqueeze(0)], 
                    dim=0
                )
                predictions[name] = torch.cat(
                    [predictions[name].unsqueeze(0), scores.unsqueeze(0)], 
                    dim=0
                )
    return features, predictions

def main_worker(config):
    # parallel setting    
    device_ids = config["parallel"]["device_ids"]
    config['parallel']['device_ids'] = device_ids

    # build dataloader
    test_loaders = build_dataloader(config['data']['test'])

    # eval projects one by one
    for proj_dir in config['project']['proj_dirs']:
        print(proj_dir)
        # load config
        config_path = osp.join(proj_dir, 'config.yml')
        with open(config_path, 'r') as f:
            test_config = yaml.load(f, yaml.SafeLoader)
    
        # build model
        bkb_net = build_from_cfg(
            test_config['model']['backbone']['net'],
            'model.backbone',
        )
        bkb_net = nn.DataParallel(bkb_net, device_ids=device_ids)
        bkb_net = bkb_net.cuda()
        bkb_net.eval()

        # model paths and run test
        model_dir = test_config['project']['model_dir']
        save_iters = test_config['project']['save_iters']
        bkb_paths = [
            osp.join(model_dir, 'backbone_{}.pth'.format(save_iter))
            for save_iter in save_iters
        ]
        features, predictions = test_run(bkb_net, bkb_paths, test_loaders)
        
        return features, predictions


if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    config['data'] = fill_config(config['data'])
 
    # override config
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.'
                'CPU mode is not supported yet')

    if args.proj_dirs:
        config['project']['proj_dirs'] = args.proj_dirs

    if args.start_time:
        yy, mm, dd, h, m, s = args.start_time.split('-')
        yy, mm, dd = int(yy), int(mm), int(dd)
        h, m, s = int(h), int(m), int(s)
        start_time = datetime(yy, mm, dd, h, m, s)
        while datetime.now() < start_time:
            print(datetime.now())
            time.sleep(600)

    main_worker(config)
