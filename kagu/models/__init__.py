import os
from kagu.models.model import BackboneArguments, Backbone, KaguModelArguments, KaguModel


def getModel(args):
    """
    Import the correct model
    """

    backbone_arguments = BackboneArguments.from_args(args)
    backbone_model = Backbone(backbone_arguments).cuda()

    kagu_arguments = KaguModelArguments.from_args(args)
    kagu_model = KaguModel(kagu_arguments).cuda()

    return backbone_model, kagu_model
    
