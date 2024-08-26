import torch, random, warnings, argparse
import numpy as np
warnings.filterwarnings("ignore")


def GetMainParameters():
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--AnnotFile', type=str, default='./data/viton_annotations.pkl')
    parser.add_argument('--DatasetRoot', type=str, default='./data/VITON_V8')
    parser.add_argument('--SaveFolder', type=str, default='./results/image')
    parser.add_argument('--RootCheckpoint', type=str, default='./results/checkpoint')
    parser.add_argument('--SplitRatio', type=float, default=0.1)
    parser.add_argument('--NumWorkers', type=int, default=8)
    parser.add_argument('--BatchSize', type=int, default=4)
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--num_head", type=int, default=4) # 3 changed to 4
    # TRAINING        
    parser.add_argument('--Optim', type=str, default="adamw")
    parser.add_argument('--Scheduler', type=str, default="")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--Seed', type=int, default=123)
    parser.add_argument('--cuda', type=bool, default=True)
    return parser.parse_args([])


def GetLossWeightParameters(args):
    # Regular G
    args.lambda_adv = 1.0
    args.lambda_skin_rm = 0.1
    return args


def GetOptimParameters(args):
    if args.Optim == 'adam':
        args.lr = 2e-4
        args.beta1 = 0.5
    elif args.Optim == 'adamw':
        args.lr = 2e-4
        args.beta1 = 0.5
    elif args.Optim == 'sgd':
        args.lr = 2e-4
        args.momentum = 0.9
    if args.Scheduler == 'cosine':
        args.T_max = 8
        args.T_mult = 2
    return args


def MetricsInit(args):
    args.best_loss = float('inf')
    return args


def SetupSeed(args):
    random.seed(args.Seed)
    np.random.seed(args.Seed)
    torch.cuda.manual_seed_all(args.Seed)

def GetModelSetting(args):
    args.channels = [64, 128, 256]
    return args

def GetConfig():
    args = GetMainParameters()
    args = GetLossWeightParameters(args)
    args = GetOptimParameters(args)
    args = MetricsInit(args)
    args = GetModelSetting(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args
