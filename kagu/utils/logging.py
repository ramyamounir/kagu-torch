import os, shutil, json
import os.path as osp


def checkdir(path, reset = True):
    if osp.exists(path):
        if reset:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

    return path


def setup_output(args):

    assert osp.exists(args.dataset)
    assert (args.dbg) or (not osp.exists(osp.join(args.output, args.name))), 'experiment logs exist'

    # create output directory
    args.exp_output = checkdir(osp.join(args.output, args.name))
    args.ckpt_dir = checkdir(f'{args.exp_output}/checkpoints')
    args.world_size = args.p_n_gpus * args.p_n_nodes

    # save commandline arguments and logs
    json.dump(args.__dict__, open(osp.join(args.exp_output, 'args.json'), 'w'), indent=2)
    args.p_logs = osp.join(args.exp_output, 'console_logs')

    return args


class TBWriter(object):

    def __init__(self, writer, data_type, tag, mul = 1, add = 0, fps = 4):

        self.step = 0
        self.mul = mul
        self.add = add
        self.fps = fps

        self.writer = writer
        self.type = data_type
        self.tag = tag

    def __call__(self, data, step = None, flush = False, metadata=None, label_img=None):

        counter = step if step != None else self.step*self.mul+self.add

        if self.type == 'scalar':
            self.writer.add_scalar(self.tag, data, global_step = counter)
        elif self.type == 'scalars':
            self.writer.add_scalars(self.tag, data, global_step = counter)
        elif self.type == 'image':
            self.writer.add_image(self.tag, data, global_step = counter)
        elif self.type == 'video':
            self.writer.add_video(self.tag, data, global_step = counter, fps = self.fps)
        elif self.type == 'figure':
            self.writer.add_figure(self.tag, data, global_step = counter)
        elif self.type == 'text':
            self.writer.add_text(self.tag, data, global_step = counter)
        elif self.type == 'histogram':
            self.writer.add_histogram(self.tag, data, global_step = counter)
        elif self.type == 'embedding':
            self.writer.add_embedding(mat=data, metadata=metadata, label_img=label_img, global_step=counter, tag=self.tag)

        self.step += 1

        if flush:
            self.writer.flush()
