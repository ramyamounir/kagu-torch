import torch, torchvision
from torchvision.utils import make_grid as makeGrid
from torchvision.transforms.functional import InterpolationMode as IM

def overlay_attention(image_tens, attention, alpha = 0.5, inter = False):

	image_tens = (image_tens - image_tens.min())/ (image_tens.max() - image_tens.min())

	if inter:
		resized = torchvision.transforms.functional.resize(attention.permute(2,0,1), (image_tens.shape[1:]), IM.BILINEAR)
	else:
		resized = torchvision.transforms.functional.resize(attention.permute(2,0,1), (image_tens.shape[1:]), IM.NEAREST)
	scaled = (resized - resized.min())/ (resized.max() - resized.min())

	overlay = (image_tens * (1-alpha)) + (scaled * alpha)

	black = torch.zeros_like(image_tens)
	hide = (black * (1-scaled)) + (image_tens * scaled)


	return makeGrid([image_tens, overlay, hide], nrow = 3)
