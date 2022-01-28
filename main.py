# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer pose estimator', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--fp16', action='store_true',
                        help='If True, Automatic Mixed Precision (AMP) is used')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    #parser.add_argument('--dilation', action='store_true',
    #                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--dilation5', action='store_true',
                        help="If true, we replace stride with dilation in the last/5th convolutional block (DC5)")
    parser.add_argument('--dilation4', action='store_true',
                        help="If true, we replace stride with dilation in the second last/4th convolutional block (DC4)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--freeze_backbone', default=False, type=bool,
                        help="If set backbone is not trained, but only extracts features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=50, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    # possible add-on
    #parser.add_argument('--deformable', action='store_false',
    #                    help="usage of deformable POET")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_kpts', default=5, type=float,
                        help="L1 kpts coefficient in the matching cost")
    parser.add_argument('--set_cost_ctrs', default=1, type=float,
                        help="L2 center kpt coefficient in the matching cost")
    parser.add_argument('--set_cost_deltas', default=1, type=float,
                        help="L1 offsets coefficient in the matching cost")
    parser.add_argument('--set_cost_kpts_class', default=2, type=float,
                        help="kpts class coefficient in the matching cost")
    
    # * Loss coefficients
    parser.add_argument('--kpts_loss_coef', default=5, type=float)
    parser.add_argument('--kpts_class_loss_coef', default=2, type=float)
    parser.add_argument('--ctrs_loss_coef', default=1, type=float)
    parser.add_argument('--deltas_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # pose estimation
    # parser.add_argument('--hierarchical', default=False, type=bool,
    #                     help="hierarchical representation of joints")
    parser.add_argument('--kpts_center', default='center_of_mass', type=str,
                        help="definition of individual's joint center")

    # dataset parameters
    parser.add_argument('--macaquepose', action='store_true',
                        help="train on macaquepose dataset")
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--max_size', default=1333, type=int,
                        help='max_size of input images when resizing')
    parser.add_argument('--coco_filter', default='',
                        help='type of annotation file to load')

    # sanity checks
    parser.add_argument('--downsampling', action='store_true',
                        help="downsample input images to iterate faster")
    parser.add_argument('--overfit', action='store_true',
                        help="train only on a small subsample of training data to overfit")

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--chkpt_save', default=20, type=int,
                        help='after how many epochs an additional checkpoint is saved')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_bb_only', action='store_true', help='only load weights from backbone')
    parser.add_argument('--resume_model_only', action='store_true',
                        help='only load weights from model, but not optimizer, lr and epochs')
    parser.add_argument('--mmpose_pre', action='store_true', help='load weights from mmpose')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # some machines have problems with cpu workers; in this case next line is needed
    #torch.multiprocessing.set_sharing_strategy('file_system')

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                        find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.freeze_backbone:
        for p in model_without_ddp.backbone.parameters():
            p.requires_grad = False
            
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("backbone" in n and p.requires_grad and not args.freeze_backbone)],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    output_dir = Path(args.output_dir)

    if args.overfit:
        # overfit network to 5 training samples
        subset_indices = np.random.choice(1000, 5, replace=False)
        sampler_train = SubsetRandomSampler(subset_indices)
        sampler_val = SubsetRandomSampler(subset_indices)
        # save subset indices
        with open(output_dir/"samples_indices.txt", "w") as text_file:
            print("{}".format(subset_indices), file=text_file)

    batch_sampler_train = torch.utils.data.BatchSampler(
                            sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.overfit: # when overfitting we want: train set = test set
        dataset_val = dataset_train
        data_loader_val = data_loader_train

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    #output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if args.resume_bb_only:
            # loading only weights of backbone
            if args.mmpose_pre:
                # other names in mmpose pretrained models param dicts
                for name, param in checkpoint['state_dict'].items():
                    if not 'backbone' in name or 'num_batches_tracked' in name:
                        continue
                    param = param.data
                    name = 'backbone.0.body' + name[len('backbone'):]
                    model_without_ddp.state_dict()[name].copy_(param)
            else:
                for name, param in checkpoint['model'].items():
                    if not 'backbone' in name:
                        continue
                    param = param.data
                    model_without_ddp.state_dict()[name].copy_(param)
        else:
            model_without_ddp.load_state_dict(checkpoint['model'])

        if not args.resume_model_only:
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        #test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
        test_stats, test_preds, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val,
                                                            base_ds, device, args.output_dir, args.fp16)
        if args.output_dir and utils.is_main_process():
            with open((output_dir / 'eval_pred.json'), 'w') as predf:
                json.dump(test_preds, predf)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["keypoints"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if args.fp16:
            # initialize gradient scaler
            scaler = torch.cuda.amp.GradScaler()
            train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, args.fp16, scaler)
        else:
            train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer,
                                          device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.chkpt_save == 0 or epoch == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:03}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        #test_stats, coco_evaluator = evaluate(
        test_stats, test_preds, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args.fp16
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with open((output_dir / 'eval_pred.json'), 'w') as predf:
                    json.dump(test_preds, predf)
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.chkpt_save == 0 or epoch == 0:
                if not os.path.exists((output_dir / 'eval')):
                    os.makedirs((output_dir / 'eval'))
                with open((output_dir / 'eval' / f'eval_pred{epoch:03}.json'), 'w') as predf:
                    json.dump(test_preds, predf)

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "keypoints" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["keypoints"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('POET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
