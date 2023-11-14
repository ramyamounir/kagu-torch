import argparse

def bool_flag(s):
    r"""
    Parse boolean arguments from the command line.

    :param str s: command line string argument
    :returns bool: the boolean value of the argument
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def parser():
    r"""
    parser function for cli arguments
    """

    parser = argparse.ArgumentParser(description='Kagu parser')

    # Paths
    parser.add_argument('--dataset', type=str, default='data/kagu', help='Path to dataset directory')
    parser.add_argument('--output', type=str, default="out", help='Path to output directory. Will be created if does not exist')
    parser.add_argument('--name', type=str, default='my_model', help='Name of the experiment for logging purposes')

    # Platform settings
    parser.add_argument('--p_name', type=str, default='job', help='Platform job name for SLURM')
    parser.add_argument('--p_device', type=str, choices=['gpu', 'slurm', 'cpu', 'mps'], default='gpu', help='Platform device')
    parser.add_argument('--p_partition', type=str, default='general', help='Platform partition for SLURM')
    parser.add_argument('--p_n_nodes', type=int, default=1, help='Platform number of nodes for SLURM')
    parser.add_argument('--p_n_gpus', type=int, default=1, help='Platform number of GPUs per node')
    parser.add_argument('--p_n_cpus', type=int, default=2, help='Platform number of total CPUs per process/GPU')
    parser.add_argument('--p_ram', type=int, default=10, help='Platform total RAM in GB')
    parser.add_argument('--p_backend', type=str, choices=['nccl', 'gloo'], default='nccl', help='Platform backend for IPC')
    parser.add_argument('--p_verbose', type=bool_flag, default=True, help='Platform verbose')
    parser.add_argument('--p_logs', type=str, default='./logs', help='Platform console logs path. Will be added to output folder automatically')


    # Dataset
    parser.add_argument('--frame_size', type=int, nargs='+', default = [299,299], help='Frame size of images')
    parser.add_argument('--snippet', type=int, default = 16, help='Number of frames to process')
    parser.add_argument('--step', type=int, default = 8, help='The stride by which we process the frames. Same as snippet if not overlapping')

    # Architecture
    parser.add_argument('--backbone', type=str, choices=["inception","resnet"], default = 'inception', help='Backbone encoder architecture')
    parser.add_argument('--backbone_pretrained', type=bool_flag, default = True, help='whether the backbone is pretrained or not')
    parser.add_argument('--backbone_frozen', type=bool_flag, default = True, help="whether the backbone's weights are frozen or not")
    parser.add_argument('--predictor', type=str, choices=["lstm","gru"], default = 'lstm', help='predictor unit')
    parser.add_argument('--dropout', type=float, default = 0.4, help='dropout rate, applied to hidden state(s)')
    parser.add_argument('--teacher', type=bool_flag, default = True, help='whether to apply teacher forcing')


    # Optimization
    parser.add_argument('--optimizer', type=str, default = 'adam', help='Optimizer function to choose')
    parser.add_argument('--lr', type=float, default = 1e-8, help='Initial Learning Rate')

    # Debugging and logging
    parser.add_argument('--dbg', action='store_true', help='Flag for debugging and development. Overrides log files.')
    parser.add_argument('--tb', action='store_true', help='Flag for tb logging. If False, does not save tensorboard files.')

    return parser

