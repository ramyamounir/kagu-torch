import torch
import torch.nn.functional as F
from kagu.utils.logging import checkdir
from kagu.utils.logging import TBWriter
from kagu.utils.vision import overlay_attention
from tqdm import tqdm
from collections import deque
import math, itertools, json
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainerArguments():
    r"""
    Arguments for Trainer class
    """

    tb_dir: str
    r""" The path to the tensorboard logging directory """

    files_dir: str
    r""" The path to the files logging directory """

    backbone: str
    r""" The backbone architecture """

    predictor: str
    r""" The predictor type """

    snippet: int
    r""" Number of frames to process """

    step: int
    r""" The stride by which we process the frames. Same as snippet if not overlapping """

    @staticmethod
    def from_args(args):
        return KaguDatasetArguments(
                tb_dir=args.tb_dir,
                files_dir=args.files_dir,
                backbone=args.predictor,
                snippet=args.snippet,
                step=args.step,
        )


class Trainer:
    r"""

    The trainer class: takes care of the training loop, logging and iterating over the dataset.

    :param TrainerArguments args: The arguments passed to Trainer class
    :param KaguDataset loader: The dataset class to be iterated over.
    :param torch.nn.Module backbone: The backbone encoder model.
    :param torch.nn.Module loss: The loss module for calculating the prediction loss.
    :param torch.nn.Module optimizer: The Adam optimizer used for stepping gradients.
    """

    def __init__(self, args:TrainerArguments, loader, backbone, model, loss, optimizer):

        self.args = args
        self.train_gen = loader
        self.backbone = backbone
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        # === TB writers === #
        self.writer = SummaryWriter(args.tb_dir)
        self.lr_writer = TBWriter(self.writer, 'scalar', 'Learning Rate Schedule')
        self.pred_L_writer = TBWriter(self.writer, 'scalar', 'Loss/prediction')
        self.motion_L_writer = TBWriter(self.writer, 'scalar', 'Loss/motion')
        self.mw_L_writer = TBWriter(self.writer, 'scalar', 'Loss/motion-weighted')
        self.attn_writer = TBWriter(self.writer, 'image', 'Attention/Bhadanau map')
        self.attn_inter_writer = TBWriter(self.writer, 'image', 'Attention/Bhadanau map BILINEAR')
        self.pred_writer = TBWriter(self.writer, 'image', 'Loss/pred map')
        self.motion_writer = TBWriter(self.writer, 'image', 'Loss/motion map')
        self.mwl_writer = TBWriter(self.writer, 'image', 'Loss/mwl map')
        self.hidden_emb_writer = TBWriter(self.writer, 'embedding', 'hidden_embedding')

        self.pred_f = open(f'{args.files_dir}/preds.txt', 'w+')
        self.attn_f = open(f'{args.files_dir}/attns.txt', 'w+')
        self.pred_grid_f = open(f'{args.files_dir}/pred_grid.txt', 'w+')
        self.mot_grid_f = open(f'{args.files_dir}/mot_grid.txt', 'w+')
        self.mwl_grid_f = open(f'{args.files_dir}/mwl_grid.txt', 'w+')


    def train(self):
        r"""
        The training function. Called once after instantiating the trainer
        """


        def get_activation(name):
            def hook(model, input, output):
                if name =='hidden':
                    activation[name].append(output[0].mean(dim=0).detach())
                    # activation[name] = output[0].mean(dim=0).detach()
                elif name == 'attention':
                    activation[name].append(output.mean(dim=0).detach())
                    # activation[name] = output.mean(dim=0).detach()
            return hook
            
        self.model.predictor.register_forward_hook(get_activation('hidden'))
        self.model.multip.register_forward_hook(get_activation('attention'))


        # Backbone
        h,w = (8,8) if self.args.backbone=="inception" else (10,10)

        # Hidden states initialization
        hx = torch.zeros(size= (h*w, 2048)).cuda()
        p = torch.zeros(size= (h*w, 2048)).cuda()

        if self.args.predictor == "lstm":
            cx = torch.zeros(size= (h*w, 2048)).cuda()
            hidden = (hx, cx)
        elif self.args.predictor == 'gru':
            hidden = hx

        write_counter = 0
        feat_deque = deque()

        for batch_idx, (input_data) in enumerate(tqdm(self.train_gen)):

            # === Inputs === #
            input_data.squeeze_(0)
            x = input_data.cuda(non_blocking=True)

            res = self.backbone(x)

            # === Forward pass === #
            feat_deque.extend(self.backbone(x).detach())

            while(len(feat_deque) > self.args.snippet):

                feats = torch.stack(list(itertools.islice(feat_deque, 0, self.args.snippet)))
                [feat_deque.popleft() for _ in range(self.args.step)]

                activation = {'hidden':[], 'attention':[]} # reset hooks
                preds, attn, hidden, p = self.model(feats, hidden, p)

                # === Loss === #
                pred, mot, mwl = self.loss(preds, feats)
                loss = mwl.mean()

                # === Sanity Check === #
                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()), force=True)
                    sys.exit(1)
                
                # === Backward pass === #
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()


                # === Logging === #
                self.lr_writer(self.optimizer.param_groups[0]['lr'])
                self.pred_L_writer(pred.mean())
                self.motion_L_writer(mot.mean())
                self.mw_L_writer(mwl.mean())
                
                # log imgs on tb
                if write_counter % (1000//self.args.step) == 0:

                    ind = self.args.step - 1
                    attn_vis = attn[ind].reshape((h,w,1)).cpu()
                    pred_vis = F.softmax(pred, dim = -1)[ind].reshape(h,w,1).cpu()
                    mot_vis = F.softmax(mot, dim = -1)[ind].reshape(h,w,1).cpu()
                    mwl_vis = F.softmax(mwl, dim = -1)[ind].reshape(h,w,1).cpu()

                    self.attn_writer(overlay_attention(input_data[ind], attn_vis), write_counter)
                    self.attn_inter_writer(overlay_attention(input_data[ind], attn_vis, inter = True), write_counter)
                    self.pred_writer(overlay_attention(input_data[ind], pred_vis), write_counter)
                    self.motion_writer(overlay_attention(input_data[ind], mot_vis), write_counter)
                    self.mwl_writer(overlay_attention(input_data[ind], mwl_vis), write_counter, flush = True)


                # log text 
                for i in range(self.args.step):

                    self.pred_f.write("{},{},{},{},{}\n".format(
                                            write_counter, 
                                            pred[i].mean().item(), 
                                            mot[i].mean().item(),
                                            mwl[i].mean().item(),
                                            str(self.optimizer.param_groups[0]['lr'])
                                            ))

                    self.attn_f.write(str(write_counter)+",")
                    [self.attn_f.write(str(ii.item())+",") for ii in attn[i].squeeze()]
                    self.attn_f.write("\n")

                    self.pred_grid_f.write(str(write_counter)+",")
                    [self.pred_grid_f.write(str(ii.item())+",") for ii in pred[i]]
                    self.pred_grid_f.write("\n")

                    self.mot_grid_f.write(str(write_counter)+",")
                    [self.mot_grid_f.write(str(ii.item())+",") for ii in mot[i]]
                    self.mot_grid_f.write("\n")

                    self.mwl_grid_f.write(str(write_counter)+",")
                    [self.mwl_grid_f.write(str(ii.item())+",") for ii in mwl[i]]
                    self.mwl_grid_f.write("\n")

                    write_counter += 1

