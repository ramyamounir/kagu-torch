import torch
from ddpw import Platform, Wrapper
from kagu.arguments.base_arguments import parser
from kagu.utils.logging import setup_output
from kagu.utils.distributed import init_gpu
import kagu.datasets as datasets
import kagu.models as models
from kagu.core.loss import Loss
from kagu.trainer.trainer import Trainer

def train_gpu(global_rank, local_rank, args):


    # initialize gpus
    init_gpu(global_rank, local_rank, args)

    # get dataloader
    loader = datasets.find_dataset_using_name(args)

    # get model
    backbone_model, kagu_model = models.getModel(args)

    # get loss
    loss = Loss()

    # get optimizer
    optimizer = torch.optim.Adam(kagu_model.parameters(), lr=args.lr)

    # get trainer
    trainer = Trainer(args, loader, backbone_model, kagu_model, loss, optimizer)
    trainer.train()


if __name__ == "__main__":

    args = parser().parse_args()
    args = setup_output(args)

    platform = Platform(
                        name=args.p_name,
                        device=args.p_device,
                        partition=args.p_partition,
                        n_nodes=args.p_n_nodes,
                        n_gpus=args.p_n_gpus,
                        n_cpus=args.p_n_cpus,
                        ram=args.p_ram,
                        backend=args.p_backend,
                        console_logs=args.p_logs,
                        verbose=args.p_verbose
                            )

    wrapper = Wrapper(platform=platform)

    # start training
    wrapper.start(train_gpu, args = args)


