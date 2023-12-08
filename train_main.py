import argparse
import time

import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

from braincog.base.node.node import *
from braincog.utils import *
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *
from braincog.model_zoo.resnet import *
from braincog.model_zoo.convnet import *
#from braincog.model_zoo.vgg_snn import VGG_SNN
from braincog.model_zoo.resnet19_snn import resnet19
from braincog.model_zoo.sew_resnet  import *
from braincog.utils import save_feature_map, setup_seed
from braincog.base.utils.visualization import *

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch import distributed as dist
from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import  NativeScaler
from RCNN import *
from torchmetrics import CalibrationError
from  torch.utils.tensorboard import SummaryWriter
from braincog.datasets import is_dvs_data
writer = None

_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='SNN Training and Evaluating')

# 模型参数
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--model', default='metarightsltet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "VGG_SNN"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')


# 静态数据集的参数
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='inputs image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')

# 数据加载器参数
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='inputs batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# 优化器参数
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='weight decay (default: 0.01 for adamw)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--adam-epoch', type=int, default=1000, help='lamb switch to adamw')

# 学习率曲线参数
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--power', type=int, default=1, help='power')

# 数据增强和正则化参数 只用于图片数据集
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.0,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--newton-maxiter', default=20, type=int,
                    help='max iterration in newton method')
parser.add_argument('--reset-drop', action='store_true', default=False,
                    help='whether to reset drop')
parser.add_argument('--kernel-method', type=str, default='cuda', choices=['torch', 'cuda'],
                    help='The implementation way of gaussian kernel method, choose from "cuda" and "torch"')

# batchnorm参数 Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between node after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# 模型指数移动平均 Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                    help='decay factor for model weights moving average (default: 0.9998)')

# 配置
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of inputs bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision training')


parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='out', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--device', type=int, default=0)

# 脉冲参数
parser.add_argument('--step', type=int, default=10, help='Simulation time step (default: 10)')
parser.add_argument('--encode', type=str, default='direct', help='Input encode method (default: direct)')
parser.add_argument('--temporal-flatten', action='store_true',
                    help='Temporal flatten to channels. ONLY FOR EVENT DATA TRAINING BY ANN')
parser.add_argument('--adaptive-node', action='store_true')
parser.add_argument('--base-step',type=int, default=0) 

# 神经元参数
parser.add_argument('--node-type', type=str, default='LIFNode', help='Node type in network (default: PLIF)')
parser.add_argument('--act-fun', type=str, default='QGateGrad',
                    help='Surogate Function in node. Only for Surrogate nodes (default: AtanGrad)')
parser.add_argument('--threshold', type=float, default=.5, help='Firing threshold (default: 0.5)')
parser.add_argument('--tau', type=float, default=2., help='Attenuation coefficient (default: 2.)')
parser.add_argument('--requires-thres-grad', action='store_true')
parser.add_argument('--sigmoid-thres', action='store_true')

parser.add_argument('--loss-fn', type=str, default='ce', help='loss function (default: ce)')
parser.add_argument('--noisy-grad', type=float, default=0.,
                    help='Add noise to backward, sometime will make higher accuracy (default: 0.)')
parser.add_argument('--spike-output', action='store_true', default=False,
                    help='Using mem output or spike output (default: False)')
parser.add_argument('--n_groups', type=int, default=1)

# 事件数据集增强
parser.add_argument('--mix-up', action='store_true', help='Mix-up for event data (default: False)')
parser.add_argument('--cut-mix', action='store_true', help='CutMix for event data (default: False)')
parser.add_argument('--event-mix', action='store_true', help='EventMix for event data (default: False)')
parser.add_argument('--cutmix_beta', type=float, default=1.0, help='cutmix_beta (default: 1.)')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix_prib for event data (default: .5)')
parser.add_argument('--cutmix_num', type=int, default=1, help='cutmix_num for event data (default: 1)')
parser.add_argument('--cutmix_noise', type=float, default=0.,
                    help='Add Pepper noise after mix, sometimes work (default: 0.)')
parser.add_argument('--gaussian-n', type=int, default=3)
parser.add_argument('--rand-aug', action='store_true',
                    help='Rand Augment for Event data (default: False)')
parser.add_argument('--randaug_n', type=int, default=3,
                    help='Rand Augment times n (default: 3)')
parser.add_argument('--randaug_m', type=int, default=15,
                    help='Rand Augment times n (default: 15) (0-30)')
parser.add_argument('--train-portion', type=float, default=0.9,
                    help='Dataset portion, only for datasets which do not have validation set (default: 0.9)')
parser.add_argument('--event-size', default=48, type=int,
                    help='Event size. Resize event data before process (default: 48)')
parser.add_argument('--layer-by-layer', action='store_true',
                    help='forward step-by-step or layer-by-layer. '
                         'Larger Model with layer-by-layer will be faster (default: False)')
parser.add_argument('--node-resume', type=str, default='',
                    help='resume weights in node for adaptive node. (default: False)')
parser.add_argument('--node-trainable', action='store_true')

# 可视化
parser.add_argument('--visualize-fp', action='store_true',
                    help='Visualize spiking map for each layer, only for validate (default: False)')
parser.add_argument('--spike-rate', action='store_true',
                    help='Print spiking rate for each layer, only for validate(default: False)')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--conf-mat', action='store_true')
parser.add_argument('--visualize-mem', action='store_true')

parser.add_argument('--suffix', type=str, default='',
                    help='Add an additional suffix to the save path (default: \'\')')

# 知识蒸馏参数
parser.add_argument('--alpha', type=float, default=1.,
                    help='1/ratio of kdloss')
parser.add_argument('--copyopt', action='store_true',
                    help='if use new opt')                    
parser.add_argument('--learner', type=str, default='VGG_SNN',
                    help='if use new opt')    
parser.add_argument('--loc', type=int, default=5,
                    help='if use new opt')        
parser.add_argument('--staticalpha', action='store_true',
                    help='if use new opt') 
parser.add_argument('--T', type=int, default=1,
                    help='if use new opt')                                     


has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

#加载参数
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args() 
    args.no_spike_output = True
    output_dir = ''
    #设置log 只有local_rank=0的进程记录
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model,
            args.dataset,
            str(args.step),
            args.suffix
            # str(args.img_size)
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        args.output_dir = output_dir
        setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))

    else:
        setup_default_logging()
    global writer
    writer=SummaryWriter(log_dir=os.path.join(output_dir, 'vis_result'))
    #判断一个进程只对应一个卡
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    #设定该进程多对应的卡
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device('cuda:%d' % args.device)
    assert args.rank >= 0
    
    # 记录log
    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    #设置种子
    setup_seed(args.seed + args.rank)
    
     
    #创建模型
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        adaptive_node=args.adaptive_node,
        dataset=args.dataset,
        step=args.step,
        encode_type=args.encode,
        node_type=eval(args.node_type),
        threshold=args.threshold,
        tau=args.tau,
        sigmoid_thres=args.sigmoid_thres,
        requires_thres_grad=args.requires_thres_grad,
        spike_output=not args.no_spike_output,
        act_fun=args.act_fun,
        temporal_flatten=args.temporal_flatten,
        layer_by_layer=args.layer_by_layer,
        n_groups=args.n_groups,
        copyopt=args.copyopt,
        learner=args.learner,
        loc=args.loc,
        cnf="ADD",
        base_step=args.base_step
        
    )

     
    #根据数据集判断通道数
    if 'dvs' in args.dataset or 'NCAL' in args.dataset:
        args.channels = 2
    elif 'mnist' in args.dataset:
        args.channels = 1
    else:
        args.channels = 3

    #设置lr

    linear_scaled_lr = args.lr * args.batch_size * args.world_size / 1024.0
    args.lr = linear_scaled_lr
    _logger.info("learning rate is %f" % linear_scaled_lr)

    #计算参数量
    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

 
    # 使用dp 在ddp模式下num_gpu已经被置1 一个进程有一个模型
    if args.num_gpu > 1:

        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model = model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    #创建优化器
    optimizer = create_optimizer(args, model)

    #使用amp
    amp_autocast = suppress
    loss_scaler = None
    if args.amp and has_native_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            if not has_native_amp:
                _logger.info('nativeAMP is not avalible')
            else:
                _logger.info('AMP not enabled. Training in float32.')

    # 是否继续训练
    resume_epoch = None
    if args.resume and args.eval_checkpoint == '':
        args.eval_checkpoint = args.resume
    if args.resume:
        #args.eval = True

        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)
        
    
    #是否记录特征
    if args.spike_rate:
        model.set_requires_fp(True)
    
    #是否使用ema
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    #是否node_resume
    if args.node_resume:
        ckpt = torch.load(args.node_resume, map_location='cpu')
        model.load_node_weight(ckpt, args.node_trainable)
    
    #转换为ddp模型 
    model_without_ddp = model
    if args.distributed:
        #是否使用sync_bn
        if args.sync_bn:
            assert not args.split_bn
            try:
 
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        #转换DDP 模型
        if args.local_rank == 0:
            _logger.info("Using native Torch DistributedDataParallel.")
        model = NativeDDP(model, device_ids=[args.local_rank],
                            find_unused_parameters=True)  # can use device str in Torch >= 1.1
        model_without_ddp = model.module
    # NOTE: EMA model does not need to be wrapped by DDP
    
    # 创建lr曲线规则
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))
    args.base_step=args.base_step if args.base_step>0 else args.step
    #获取数据集
    # now config only for imnet
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    loader_train, loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % args.dataset)(
        batch_size=args.batch_size,
        step=args.base_step,
        args=args,
        _logge=_logger,
        data_config=data_config,
        num_aug_splits=num_aug_splits,
        size=args.event_size,
        mix_up=args.mix_up,
        cut_mix=args.cut_mix,
        event_mix=args.event_mix,
        beta=args.cutmix_beta,
        prob=args.cutmix_prob,
        gaussian_n=args.gaussian_n,
        num=args.cutmix_num,
        noise=args.cutmix_noise,
        num_classes=args.num_classes,
        rand_aug=args.rand_aug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        portion=args.train_portion,
        _logger=_logger,
    )

    # 设置损失函数
    if args.loss_fn == 'mse':
        train_loss_fn = UnilateralMse(1.)
        validate_loss_fn = UnilateralMse(1.)
    elif args.loss_fn == 'mix':
        train_loss_fn = MixLoss(train_loss_fn)
        validate_loss_fn = MixLoss(validate_loss_fn)    
    else:
        if args.jsd:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        elif mixup_active:
            # smoothing is handled with mixup target transform
            train_loss_fn = SoftTargetCrossEntropy().cuda()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        else:
            train_loss_fn = nn.CrossEntropyLoss().cuda()

        validate_loss_fn = nn.CrossEntropyLoss().cuda()



    #结果统计 metric
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None

    #如果只进行测试
    if args.eval:  # evaluate the model
        if args.distributed:
            state_dict = torch.load(args.eval_checkpoint)['state_dict_ema']
            new_state_dict = OrderedDict()
            # add module prefix for DDP
            for k, v in state_dict.items():
                k = 'module.' + k
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
        else:
             load_checkpoint(model, args.eval_checkpoint, args.model_ema)
             
             
             
        for i in range(1):
            val_metrics = validate(start_epoch, model, loader_eval, validate_loss_fn, args)
            print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.3f}%")
        return

    #保存权重checkpoint的设置
    saver = None
    if args.local_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    #训练模型
    try:  # train the model
        # 重置drop
        if args.reset_drop:
            model_without_ddp.reset_drop_path(0.0)

        # epoch循环 从start_rpoch到总epoch
        for epoch in range(start_epoch, args.epochs):
            if epoch == 0 and args.reset_drop:
                model_without_ddp.reset_drop_path(args.drop_path)
            
            #同步所有进程的种子 dataloder的
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            #训练一个epoch
            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
            
            #设置分布式的bn
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            
            #测试模型
            eval_metrics = validate(epoch, model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            #ema重新测试一遍
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    epoch, model_ema.ema, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics
            
            # 更新lr策略
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
            
            # 更新统计结果 写csv的
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            # 保存结果
            # if saver is not None and epoch >= args.n_warm_up:
            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
            #画出各种参数
            if args.local_rank == 0:
                writer.add_scalars(main_tag="acc_curve",tag_scalar_dict={  "train_top1":train_metrics["top1"],"val_top1":eval_metrics["top1"],
                                                                        "train_top5":train_metrics["top5"],"val_top5":eval_metrics["top5"]},global_step=epoch)
                writer.add_scalar(tag="acc_curve/train_top1",scalar_value=train_metrics["top1"],global_step=epoch)
                writer.add_scalar(tag="acc_curve/train_top5",scalar_value=train_metrics["top5"],global_step=epoch)
                writer.add_scalar(tag="acc_curve/val_top1",scalar_value=eval_metrics["top1"],global_step=epoch)
                writer.add_scalar(tag="acc_curve/val_top5",scalar_value=eval_metrics["top5"],global_step=epoch)

                writer.add_scalars(main_tag="loss_curve",tag_scalar_dict={"train":train_metrics["loss"],"val":eval_metrics["loss"]},global_step=epoch)
                writer.add_scalar(tag="loss_curve/train",scalar_value=train_metrics["loss"],global_step=epoch)
                writer.add_scalar(tag="loss_curve/val",scalar_value=eval_metrics["loss"],global_step=epoch)
 
                
                
    except KeyboardInterrupt:
        pass
    #用于跑完或者中途中断后显示最好结果
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    while 1:
        i=1
 
class Custom_CrossEntropy_PSKD(nn.Module):
    def __init__(self,T=1):
        super(Custom_CrossEntropy_PSKD, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.T=T
    def forward(self, output, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        T=self.T
        targets=F.softmax(targets/T,dim=1)
        log_probs = self.logsoftmax(output)
        loss = T*T*(- targets * log_probs).mean(0).sum()
        return loss
def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    # 数据增强的调整
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    #切换训练模式 
    model.train()
    softloss_fn=Custom_CrossEntropy_PSKD(T=args.T)
      

     
    end = time.time()
    #最后一个iter的位置
    last_idx = len(loader) - 1
    #更新的次数 也就是第几次更新
    num_updates = epoch * len(loader)
    
    if args.staticalpha:current_alpha=args.alpha
    else:current_alpha=1-epoch/args.epochs*args.alpha if  epoch<args.epochs else 1-args.alpha
    #开始训练每个iter
    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        #开始计时
        data_time_m.update(time.time() - end)

        if not args.prefetcher or args.dataset != 'imnet':
            inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
            if mixup_fn is not None:
                inputs, target = mixup_fn(inputs, target)
        #将数据变成通道在后面模式
        if args.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        
        #前传
        with amp_autocast():
            output,loss1,loss2 = model(inputs,target=target, loss_fn=loss_fn,softloss_fn=softloss_fn )
            loss = current_alpha*loss1+(1-current_alpha)*loss2
            
        #计算准确率
        if not (args.cut_mix | args.mix_up | args.event_mix) and args.dataset != 'imnet':
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = torch.tensor([0.]), torch.tensor([0.])

        
        #更新各种参数记录
        spike_rate_avg_layer_str = ''
        threshold_str = ''
        if not args.distributed:
            losses_m.update(loss.item(), inputs.size(0))#记录loss
            top1_m.update(acc1.item(), inputs.size(0))#记录top1
            top5_m.update(acc5.item(), inputs.size(0))#记录top5

            spike_rate_avg_layer = model.get_fire_rate().tolist()
            spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
            threshold = model.get_threshold()
            threshold_str = ['{:.3f}'.format(i) for i in threshold]

        #清理梯度
        optimizer.zero_grad()
        #设置loss_scaler
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)
        else:
            #反向传播
            loss.backward(create_graph=second_order)
            #是否给grad加噪音
            if args.noisy_grad != 0.:
                random_gradient(model, args.noisy_grad)
            #是否截断梯度
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            #更新
            if args.opt == 'lamb':
                optimizer.step(epoch=epoch)
            else:
                optimizer.step()
        #同步cuda 用于计时
        torch.cuda.synchronize()
        #ema更新
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1
            
        #记录本次iter时间
        batch_time_m.update(time.time() - end)
        
        #打印各种参数记录 每log_interval来一次
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            

            #记录loss
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), inputs.size(0))
            #打印各种参数
            if args.local_rank == 0:
                if args.distributed:
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m,
                            rate=inputs.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m
                        ))
                else:
                    _logger.info(
                        'Train:{} [{:>3d}/{}({:>3.0f}%)] '
                        'Loss:{loss.val:>9.6f}({loss.avg:>6.4f}) '
                        'Acc@1:{top1.val:>7.4f}({top1.avg:>7.4f}) '
                        'Acc@5:{top5.val:>7.4f}({top5.avg:>7.4f})\n'
                        'Time:{batch_time.val:.3f}s,{rate:>7.2f}/s '
                        '({batch_time.avg:.3f}s,{rate_avg:>7.2f}/s) '
                        'Datatime: {data_time.val:.3f} ({data_time.avg:.3f})'
                        'LR:{lr:.3e}\n'
                        'Fire_rate: {spike_rate}\n'
                        'Thres: {threshold}\n'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m,
                            rate=inputs.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                            data_time=data_time_m,
                            lr=lr,
                            spike_rate=spike_rate_avg_layer_str,
                            threshold=threshold_str,
                            
                        ))
                #保存输入图片 
                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        inputs,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        # 用于恢复继续训练的权重保存
        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)
        #lr策略更新
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
    # end for
    if args.local_rank == 0:
        writer.add_scalar(tag="lr",scalar_value=lr,global_step=num_updates)
                
    #干嘛的
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    #返回loss
    return OrderedDict([('loss', losses_m.avg),("top1",top1_m.avg),("top5",top5_m.avg)])


def validate(epoch, model, loader, loss_fn, args, amp_autocast=suppress,log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    output_gather=TensorGather()
    target_gather=TensorGather()
    feature_vec = TensorGather()
 
    mem_vec = None
    ece=CalibrationError().cuda()
    #切换成测试模式
    model.eval()

    


    #开始计时
    end = time.time()
    last_idx = len(loader) - 1
    #测试每一个iter
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
             
            last_batch = batch_idx == last_idx
            #将数据放到cuda
            if not args.prefetcher or args.dataset != 'imnet':
                inputs = inputs.type(torch.FloatTensor).cuda()
                target = target.cuda()
            #设置通道在后面
            if args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            #设置需要可视化特征
            if not args.distributed:
                if args.visualize_fp or args.spike_rate or args.tsne or args.conf_mat:
                    model.set_requires_fp(True)
                if args.visualize_mem:
                    model.set_requires_mem(True)
            #前传
            with amp_autocast():
                output = model(inputs)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            #损失函数
            loss = loss_fn(output, target)
            
                 
            acc1, acc5 = accuracy(output, target , topk=(1, 5))
            curclass=0
 

           
            # 合并结果
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
                
                
            else:
                reduced_loss = loss.data
                
                output_gather.update( output.detach())
                target_gather.update( target.detach())
                

            #用于同步计时
            torch.cuda.synchronize()
            
            #记录结果
            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0)) 
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            ece.update(output,target)

            
            #用于可视化的记录
            if not args.distributed:
                #记录特征图
                if args.visualize_fp:
                    x = model.get_fp()
                    feature_path = os.path.join(args.output_dir, 'feature_map')
                    if os.path.exists(feature_path) is False:
                        os.mkdir(feature_path)
                    save_feature_map(x, feature_path)

                #打印tsne
                if args.tsne:
                    x = model.get_fp(temporal_info=False)[-1]
                    #x = torch.nn.AdaptiveAvgPool2d((1, 1))(x)
                    x = x.reshape(x.shape[0], -1)
                    feature_vec.update(x.detach())


                #记录脉冲发放率
                if args.spike_rate:
                    avg, var, spike, avg_per_step = model.get_spike_info()
                    save_spike_info(
                        os.path.join(args.output_dir, 'spike_info.csv'),
                        epoch, batch_idx,
                        args.step, avg, var,
                        spike, avg_per_step)
                #记录膜电势
                if args.visualize_mem:
                    mem_vec = model.get_mem(temporal_info=False)#[l,[t,[b,w,h]]]
                     


                #关于神经元的记录
                threshold_str = ['{:.3f}'.format(i) for i in model.get_threshold()]
                spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in model.get_fire_rate().tolist()]
                tot_spike = model.get_tot_spike()

            #打印结果
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix

                if args.distributed:
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})'
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name,
                            batch_idx,
                            last_idx,
                            batch_time=batch_time_m,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m,
                            ))
                else:
                    _logger.info(
                        '{0}:[{1:>3d}/{2}] '
                        'Loss:{loss.val:>7.4f}({loss.avg:>6.4f}) '
                        'Acc@1:{top1.val:>7.4f}({top1.avg:>7.4f}) '
                        'Acc@5:{top5.val:>7.4f}({top5.avg:>7.4f})\n'
                        'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                        'Fire_rate:{spike_rate}\n'
                        'Tot_spike:{tot_spike}\n'
                        'Thres:{threshold}\n'.format(
                            log_name,
                            batch_idx,
                            last_idx,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m,
                            spike_rate=spike_rate_avg_layer_str,
                            tot_spike=tot_spike,
                            threshold=threshold_str,

                        ))
    #end for 
    aurc_value=0
    ece_value=0
    if args.local_rank == 0 and not args.distributed:
        ece_value=ece.compute()
        aurc_value=calc_aurc(output_gather.gather,target_gather.gather)[0]*1000
        writer.add_pr_curve(tag="pr_curve/val",labels=F.one_hot(target_gather.gather,args.num_classes),predictions=F.softmax(output_gather.gather,dim=1),global_step=epoch)
    
        writer.add_scalar(tag="ece",scalar_value=ece_value,global_step=epoch)
        writer.add_scalar(tag="aurc",scalar_value=aurc_value,global_step=epoch)
        
        if args.visualize_mem: 
            for t in range(len(mem_vec )):
                
                for i  in range(len(mem_vec[t])):
                    writer.add_histogram(tag="mem/layer"+str(t),values=mem_vec[t][i],global_step=i)
                     
                    writer.add_histogram(tag="mem/time"+str(i),values=mem_vec[t][i],global_step=t)
                    
                
            
                
    
        if args.tsne:
 
            plot_tsne(feature_vec.gather, target_gather.gather,num_classes=args.num_classes, output_dir=os.path.join(args.output_dir, 't-sne-2d.eps'))
            plot_tsne_3d(feature_vec.gather, target_gather.gather, num_classes=args.num_classes,output_dir=os.path.join(args.output_dir, 't-sne-3d.eps'))

        if args.conf_mat:

            plot_confusion_matrix(output_gather.gather, target_gather.gather, os.path.join(args.output_dir, 'confusion_matrix.eps'))

        if 0:
            
           
            plot_mem_distribution(data=mem,output_dir=os.path.join(args.output_dir, 'mem_distribution.eps'),)

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg),("aurc",aurc_value),("ece",ece_value)])
    return metrics

 
if __name__ == '__main__':
    main()