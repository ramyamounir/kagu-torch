import torch
import torch.nn as nn

class Loss(nn.Module):
    r"""
    MSE Loss function is applied to minimize the distance between the prediction and actual input.
    This MSE loss is applied on the whole snippet.
    """

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.MSELoss(reduction = 'none')

    def forward(self, pred, label):
        r"""
        The loss calculation in the forward function

        :param torch.Tensor pred: The predicted features of the model
        :param torch.Tensor label: The actual inputs features
        :returns: 
            * (*torch.Tensor*): The prediction loss
            * (*torch.Tensor*): The motion loss.
            * (*torch.Tensor*): The motion weighted loss
        """

        pred_loss = self.loss_fn(pred, label[1:])
        mot_loss = torch.stack([self.loss_fn(label[i], label[i+1]) for i in range(len(label)-1)])
        mwl_loss = torch.mul(pred_loss,mot_loss)

        return pred_loss.mean(dim=-1), mot_loss.mean(dim=-1), mwl_loss.mean(dim=-1)
