import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

from collections import deque
from glob import glob
import cv2, sys
from dataclasses import dataclass

@dataclass
class KaguDatasetArguments():
    r"""
    Arguments for Kagu dataset
    """

    dataset: str
    r""" The path to the datset """

    frame_size: list
    r""" The frame size of the images """

    world_size: int
    r"""The number of proceses spawned """

    global_rank: int
    r"""The rank of the current process """

    snippet: int
    r""" Number of frames to process """

    step: int
    r""" The stride by which we process the frames. Same as snippet if not overlapping """

    @staticmethod
    def from_args(args):
        return KaguDatasetArguments(
                dataset=args.dataset,
                frame_size=args.frame_size,
                world_size=args.world_size,
                global_rank=args.global_rank,
                snippet=args.snippet,
                step=args.step,
        )


class KaguDataset(Dataset):
    r"""
    The kagu dataset class that streams video files and outputs a list of frames according to snippet and step.

    :param KaguDatasetArguments args: The parameters used for the Kagu dataset
    """

    def __init__(self, args: KaguDatasetArguments):
        self.args = args

        # override snippet and step to use backbone
        args.snippet = 40
        args.step = 40

        self.TT = T.Compose([ 
                    T.ConvertImageDtype(torch.float32), 
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                    T.Resize(args.frame_size)])


        self.frames = deque(maxlen = args.snippet)

        vids = sorted(glob('/'.join([args.dataset,'v_*.mp4'])))

        vids_chunk = torch.chunk(torch.arange(len(vids)), args.world_size)[args.global_rank]
        self.start_frame = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor([36_000]*len(vids)), 0)[:-1]])[vids_chunk[0]]
        self.end_frame = torch.cumsum(torch.tensor([36_000]*len(vids)), 0)[vids_chunk[-1]]
        self.vids = vids[vids_chunk[0]:vids_chunk[-1]+1]
        self.len = (36_000 * len(self.vids)) // args.step

        self.current_vid = 0
        self.cap = cv2.VideoCapture(self.vids[self.current_vid])

    def __getitem__(self, index):
        r"""
        Iterates over the dataset in a streaming fashion and retrieves :py:class:`KaguDatasetArguments.snippet` frames at a time.

        :param int index: The index of the item in the dataset to retrieve. Not used.
        :returns:
            * (*torch.tensor*): the frames in tensor format
        """

        # Fill deque
        while len(self.frames) < self.args.snippet:
            ret, frame = self.cap.read()

            if ret == False:

                self.current_vid += 1

                if self.current_vid == len(self.vids):
                    sys.exit('Dataset finished')

                self.cap = cv2.VideoCapture(self.vids[self.current_vid])
            else:
                frame_tensor = self.TT(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2,0,1))

                self.frames.append(frame_tensor)

        # Take snippet
        images = torch.stack(list(self.frames))

        # pop step from left
        [self.frames.popleft() for _ in range(self.args.step)]
        
        return images

    def __len__(self):
        return self.len

