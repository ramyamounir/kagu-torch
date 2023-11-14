import torch
import numpy as np, os, random
import torch.backends.cudnn as cudnn
from kagu.utils.logging import checkdir


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_gpu(global_rank, local_rank, args):

    # Initialize distributed environment
    args.local_rank = local_rank
    args.global_rank = global_rank
    args.device = torch.device("cuda:{}".format(args.local_rank))
    torch.cuda.set_device(args.device)

    fix_random_seeds()
    cudnn.benchmark = True

    args.main = (args.global_rank == 0)
    if args.p_device != 'slurm': setup_for_distributed(args.main)

    # Tensorboard logger
    args.logger = None
    # if args.tb:
    #     logger_class = loggers.getLogger(args)
    #     logger_path = os.path.join(args.exp_output, 'logs', str(args.global_rank))
    #     args.logger = logger_class(logger_path, args.log_base_every, args.distance_mode)

    args.tb_dir = checkdir(f'{args.exp_output}/logs/{args.global_rank}')
    args.files_dir = checkdir(f'{args.exp_output}/files/{args.global_rank}')


