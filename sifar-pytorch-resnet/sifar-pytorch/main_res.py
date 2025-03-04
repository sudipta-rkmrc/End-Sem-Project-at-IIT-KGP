import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import warnings

from pathlib import Path

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from engine import train_one_epoch, evaluate
from samplers import RASampler
import torch.nn as nn
import utils
from losses import DeepMutualLoss, ONELoss, MulMixturelLoss, SelfDistillationLoss

from video_dataset import VideoDataSet, VideoDataSetLMDB, VideoDataSetOnline
from video_dataset_aug import get_augmentor, build_dataflow
from video_dataset_config import get_dataset_config, DATASET_CONFIG

warnings.filterwarnings("ignore", category=UserWarning)
#torch.multiprocessing.set_start_method('spawn', force=True)

# Instead of using create_model, import our custom ResNet50 builder.
from my_models.sifar_resnet import build_resnet50

def get_args_parser():
    parser = argparse.ArgumentParser('ResNet-50 training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', default='st2stv2',
                        choices=list(DATASET_CONFIG.keys()), help='dataset name')
    parser.add_argument('--duration', default=8, type=int, help='number of frames per clip')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='number of frames per group')
    parser.add_argument('--threed_data', action='store_true',
                        help='load data in the layout for 3D conv')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop, directly crop to input_size')
    parser.add_argument('--random_sampling', action='store_true',
                        help='perform deterministic sampling for data loader')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='data augmentation version')
    parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])
    parser.add_argument('--use_lmdb', action='store_true', help='use lmdb instead of jpeg')
    parser.add_argument('--use_pyav', action='store_true', help='use video directly')

    # Model & Temporal parameters (for transformer models, not used by ResNet)
    parser.add_argument('--pretrained', action='store_true', default=False,
                    help='start with pretrained model')
    # These are ignored for ResNet, but still parsed:
    parser.add_argument('--hpe_to_token', action='store_true', help='(ignored for ResNet)')
    parser.add_argument('--rel_pos', action='store_true', help='(ignored for ResNet)')
    parser.add_argument('--window_size', default=7, type=int, help='(ignored for ResNet)')
    # Update help to indicate this parameter is used for super image creation.
    parser.add_argument('--super_img_rows', default=3, type=int, 
                        help='Number of rows to arrange frames into a super image (if >1, video frames are rearranged into a grid)')

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='model name (for logging purposes)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='EMA decay')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='Force EMA on CPU')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound (default: 1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='AutoAugment policy (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase probability (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first augmentation split')

    # Mixup parameters
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup/cutmix (default: 1.0)')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both are enabled (default: 0.5)')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='Mixup mode: "batch", "pair", or "elem" (default: "batch")')

    # Output and general training settings
    parser.add_argument('--output_dir', default='',
                        help='Path to save outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device for training/testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--no-resume-loss-scaler', action='store_false', dest='resume_loss_scaler')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable AMP')
    parser.add_argument('--use_checkpoint', default=False, action='store_true', help='Use checkpoint to save memory')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    parser.add_argument('--num_clips', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training setup')
    parser.add_argument('--auto-resume', action='store_true', help='Auto resume')
    parser.add_argument('--simclr_w', type=float, default=0., help='Weight for SimCLR loss')
    parser.add_argument('--contrastive_nomixup', action='store_true', help='Do not use mixup in contrastive learning')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for NCE')
    parser.add_argument('--branch_div_w', type=float, default=0., help='Weight for branch divergence loss')
    parser.add_argument('--simsiam_w', type=float, default=0., help='Weight for SimSiam loss')
    parser.add_argument('--moco_w', type=float, default=0., help='Weight for MoCo loss')
    parser.add_argument('--byol_w', type=float, default=0., help='Weight for BYOL loss')
    parser.add_argument('--finetune', action='store_true', help='Finetune model')
    parser.add_argument('--initial_checkpoint', type=str, default='', help='Path to pretrained model')
    parser.add_argument('--dml_w', type=float, default=0., help='Deep mutual learning weight')
    parser.add_argument('--one_w', type=float, default=0., help='ONE weight')
    parser.add_argument('--kd_temp', type=float, default=1.0, help='Temperature for KD loss')
    parser.add_argument('--mulmix_b', type=float, default=0., help='MulMixture beta')
    parser.add_argument('--hard_contrastive', action='store_true', help='Use hard contrastive loss')
    parser.add_argument('--selfdis_w', type=float, default=0., help='Self distillation weight')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    # Patch
    if not hasattr(args, 'hard_contrastive'):
        args.hard_contrastive = False
    if not hasattr(args, 'selfdis_w'):
        args.selfdis_w = 0.0

    device = torch.device(args.device)

    # Fix seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset, args.use_lmdb
    )
    args.num_classes = num_classes
    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    print(f"Creating model: {args.model}")
    
    # Build the ResNet-50 model with super image creation.
    model = build_resnet50(num_classes=args.num_classes, duration=args.duration, 
                           super_img_rows=args.super_img_rows, img_size=args.input_size)
    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.dml_w > 0.:
        criterion = DeepMutualLoss(criterion, args.dml_w, args.kd_temp)
    elif args.one_w > 0.:
        criterion = ONELoss(criterion, args.one_w, args.kd_temp)
    elif args.mulmix_b > 0.:
        criterion = MulMixturelLoss(criterion, args.mulmix_b)
    elif args.selfdis_w > 0.:
        criterion = SelfDistillationLoss(criterion, args.selfdis_w, args.kd_temp)

    max_accuracy = 0.0
    output_dir = Path(args.output_dir)

    if args.initial_checkpoint:
        print("Loading pretrained model")
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        utils.load_checkpoint(model, checkpoint['model'])

    if args.auto_resume:
        if args.resume == '':
            args.resume = str(output_dir / "checkpoint.pth")
            if not os.path.exists(args.resume):
                args.resume = ''

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        utils.load_checkpoint(model, checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint and args.resume_loss_scaler:
                print("Resume with previous loss scaler state")
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            max_accuracy = checkpoint['max_accuracy']

    mean = (0.5, 0.5, 0.5) if 'mean' not in model.default_cfg else model.default_cfg['mean']
    std = (0.5, 0.5, 0.5) if 'std' not in model.default_cfg else model.default_cfg['std']

    if args.use_lmdb:
        video_data_cls = VideoDataSetLMDB
    elif args.use_pyav:
        video_data_cls = VideoDataSetOnline
    else:
        video_data_cls = VideoDataSet
    train_list = os.path.join(args.data_dir, train_list_name)

    train_augmentor = get_augmentor(True, args.input_size, mean, std, threed_data=args.threed_data,
                                    version=args.augmentor_ver, scale_range=args.scale_range, dataset=args.dataset)
    dataset_train = video_data_cls(args.data_dir, train_list, args.duration, args.frames_per_group,
                                   num_clips=args.num_clips,
                                   modality=args.modality, image_tmpl=image_tmpl,
                                   dense_sampling=args.dense_sampling,
                                   transform=train_augmentor, is_train=True, test_mode=False,
                                   seperator=filename_seperator, filter_video=filter_video)
    

    num_tasks = utils.get_world_size()
    data_loader_train = build_dataflow(dataset_train, is_train=True, batch_size=args.batch_size,
                                       workers=args.num_workers, is_distributed=args.distributed)
    
 


    val_list = os.path.join(args.data_dir, val_list_name)
    val_augmentor = get_augmentor(False, args.input_size, mean, std, args.disable_scaleup,
                                  threed_data=args.threed_data, version=args.augmentor_ver,
                                  scale_range=args.scale_range, num_clips=args.num_clips, num_crops=args.num_crops, dataset=args.dataset)
    dataset_val = video_data_cls(args.data_dir, val_list, args.duration, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=val_augmentor, is_train=False, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    data_loader_val = build_dataflow(dataset_val, is_train=False, batch_size=args.batch_size,
                                     workers=args.num_workers, is_distributed=args.distributed)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=False, amp=args.amp, num_crops=args.num_crops, num_clips=args.num_clips)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training, current max accuracy is {max_accuracy:.2f}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, num_tasks, True,
            amp=args.amp,
            # Uncomment and pass additional loss criteria if needed:
            # simclr_criterion=simclr_criterion, simclr_w=args.simclr_w,
            # branch_div_criterion=branch_div_criterion, branch_div_w=args.branch_div_w,
            # simsiam_criterion=simsiam_criterion, simsiam_w=args.simsiam_w,
            # moco_criterion=moco_criterion, moco_w=args.moco_w,
            # byol_criterion=byol_criterion, byol_w=args.byol_w,
            contrastive_nomixup=args.contrastive_nomixup,
            hard_contrastive=args.hard_contrastive,
            finetune=args.finetune
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device, num_tasks, distributed=False, amp=args.amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if test_stats["acc1"] == max_accuracy:
                checkpoint_paths.append(output_dir / 'model_best.pth')
            for checkpoint_path in checkpoint_paths:
                state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'scaler': loss_scaler.state_dict(),
                    'max_accuracy': max_accuracy
                }
                if args.model_ema:
                    state_dict['model_ema'] = get_state_dict(model_ema)
                utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
