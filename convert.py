import argparse
import os
from os import path
import pathlib

import torch
from torch.nn import functional as F
from fastai.vision import models

from fasterai import generators
from fasterai import dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        required=True,
    )
    parser.add_argument(
        '--output',
        default='model.pth',
    )
    parser.add_argument(
        '--artistic',
        default=False,
        action='store_true'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder = pathlib.Path(path.dirname(args.path))
    name = path.basename(args.path)
    if name.endswith('.pth'):
        name = name[:-4]
    if not args.artistic:
        data = dataset.get_dummy_databunch()
        learn = generators.gen_learner_wide(
            data=data, gen_loss=F.l1_loss, nf_factor=2, arch=models.resnet101
        )
        learn.path = folder
        learn.model_dir = '.'
        learn.load(name)
    else:
        data = dataset.get_dummy_databunch()
        learn = generators.gen_learner_deep(
            data=data, gen_loss=F.l1_loss, arch=models.resnet34, nf_factor=1.5
        )
        learn.path = folder
        learn.model_dir = '.'
        learn.load(name)

    learn.model.eval()

    output_dir = path.dirname(args.output)
    if output_dir and not path.exists(output_dir):
        os.mkdir(output_dir)

    torch.save(learn.model, args.output)
    print('Saved to %s' % args.output)


if __name__ == '__main__':
    main()
