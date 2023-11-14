import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from dataclasses import dataclass


@dataclass
class BackboneArguments():
    r"""
    Arguments for the Backbone
    """

    backbone: str
    r""" The backbone architecture """

    backbone_pretrained: bool
    r""" Use pretrained backbone weights """

    backbone_frozen: bool
    r""" Freeze the backbone weights """

    @staticmethod
    def from_args(args):
        return BackboneArguments(
                backbone = args.backbone,
                backbone_pretrained=args.backbone_pretrained, 
                backbone_frozen=args.backbone_frozen,
                )


class Backbone(nn.Module):
    r"""
    The backbone encoder model.

    This model receives input images and returns feature vector grid.

    :param BackboneArguments args: The parameters used for the Backbone Model
    """

    def __init__(self, args:BackboneArguments):
        super().__init__()

        self.args = args

        # ====== Define Backbone ====== #
        if args.backbone == "resnet":
            if args.backbone_pretrained:
                self.encoder = nn.Sequential(*list(torchvision_models.__dict__["resnet50"](weights=torchvision_models.Resnet50_Weights.IMAGENET1K_V1).children())[0:8])
            else:
                self.encoder = nn.Sequential(*list(torchvision_models.__dict__["resnet50"]().children())[0:8])

        elif args.backbone == "inception":
            if args.backbone_pretrained:
                self.encoder = torchvision_models.__dict__["inception_v3"](weights=torchvision_models.Inception_V3_Weights.IMAGENET1K_V1)
                self.encoder.avgpool = nn.Sequential()
                self.encoder.dropout = nn.Sequential()
                self.encoder.fc = nn.Sequential()
            else:
                self.encoder = nn.Sequential(*list(torchvision_models.__dict__["inception_v3"]().children())[:-3])
                self.encoder.avgpool = nn.Sequential()
                self.encoder.dropout = nn.Sequential()
                self.encoder.fc = nn.Sequential()
        
        self.encoder.eval()
        if args.backbone_frozen: self.encoder.requires_grad_(False)

    def forward(self, x):
        r"""
        The forward propagation function that takes input image and returns a grid of output vectors

        :param torch.Tensor x: tensor of shape [Snippet, 3, H, W]
        :returns:
            * (*torch.Tensor*): feature vector of shape [Snippet, h*w, 2048]
        """

        # ====== BACKBONE ====== #
        S,C,H,W = x.shape
        net = self.encoder(x)
        net = net.reshape((S, 2048, -1)).permute((0,2,1))

        return net

# wrapping the multiplication operation in nn.Module to assign forward hook
class Multip(nn.Module):
    def __init__(self):
        super(Multip, self).__init__()    
    def forward(self, A, B):
        return torch.mul(A,B)

# predictor, dropout, teacher, step

@dataclass
class KaguModelArguments():
    r"""
    Arguments for the Kagu model
    """

    predictor: str
    r""" The predictor type """

    dropout: float
    r""" The dropout rate """

    teacher: bool
    r""" Enable Teacher forcing """

    step: int
    r""" The stride by which the dataset is streamed """
    
    @staticmethod
    def from_args(args):
        return KaguModelArguments(
                predictor = args.predictor,
                dropout=args.dropout, 
                teacher=args.teacher,
                step=args.step,
                )


class KaguModel(nn.Module):
    r"""
    The Implementation of the Kagu model for training.

    :param KaguModelArguments args: The arguments passed to the Kagu Model
    """

    def __init__(self, args: KaguModelArguments):
        super().__init__()
        self.args = args
        

        # ====== Define Predictor ====== #
        if args.predictor == "lstm":
            self.predictor = nn.LSTMCell(2048, 2048)
        elif args.predictor == "gru":
            self.predictor = nn.GRUCell(2048, 2048)

        # ====== Define Layers ====== #
        self.w1 = nn.Linear(2048,2048)
        self.w2 = nn.Linear(2048,2048)
        self.v = nn.Linear(2048,1)

        self.w3 = nn.Linear(4096,2048)
        self.w4 = nn.Linear(2048,2048)

        self.drop = nn.Dropout(args.dropout)
        self.multip = Multip()

        # self.apply(self._init_weights)

    def forward(self, x, hidden, p):
        r"""
        The forward propagation function of the main Kagu model.

        :param torch.Tensor x: tensor of shape [snippet, h*w, 2048]
        :param tuple(torch.Tensor, torch.Tensor) hidden: The hidden state is a tuple of tensors (each of size [h*w, 2048]) if predictor is LSTM
        :param torch.Tensor p: previous prediction for teacher forcing. Shape [h*w, 2048]
        :returns:
            * (*torch.Tensor*): Prediction of the model [snippet, h*w, 2048]
            * (*torch.Tensor*): attention grid [8, h*w, 1]
            * (*tuple(torch.Tensor, torch.Tensor)*): hidden states of the LSTM.
            * (*torch.Tensor*): P_out for teacher forcing
        """

        # ====== PREDICTION HEAD ====== #
        preds = []
        attns = []

        if self.args.predictor == "lstm":
            hx, cx = hidden
        elif self.args.predictor == "gru":
            hx = hidden

        for i, n in enumerate(x):

            # === ATTENTION === #
            attn = torch.tanh(self.w1(n) + self.w2(hx))
            attn_SM = F.softmax(self.v(attn), dim = 0)
            n_weighted = self.multip(n, attn_SM)

            if self.args.teacher:
                concat_vector = self.w3(torch.cat([n_weighted, n], dim = -1))
            else:
                concat_vector = self.w3(torch.cat([n_weighted, p], dim = -1))
            
            # === Predictor === #

            if self.args.predictor == "lstm":
                (hx, cx) = self.predictor(concat_vector, hx = (self.drop(hx), self.drop(cx)) )
                
            elif self.args.predictor == "gru":
                hx = self.predictor(concat_vector, hx = self.drop(hx) )

            # Prediction layer
            p = self.w4(hx)
            preds.append(p)
            
            # Attention maps
            if i < self.args.step:
                attns.append(attn_SM)
            
            # States output
            if i == (self.args.step-1):
                hx_out = hx.detach()
                p_out = p.detach()

                if self.args.predictor == "lstm":
                    cx_out = cx.detach()


        attns = torch.stack(attns)
        preds = torch.stack(preds[:-1])

        if self.args.predictor == "lstm":
            hidden = (hx_out, cx_out)
        elif self.args.predictor == "gru":
            hidden = hx_out


        return preds, attns.detach(), hidden, p_out

    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.LSTMCell) or isinstance(m, nn.GRUCell):
            torch.nn.init.xavier_uniform_(m.weight_ih)
            torch.nn.init.xavier_uniform_(m.weight_hh)
            if m.bias is not None:
                nn.init.constant_(m.bias_ih, 0)
                nn.init.constant_(m.bias_hh, 0)
