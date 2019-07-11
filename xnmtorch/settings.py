import random
from argparse import ArgumentParser

import numpy as np
import torch

LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_REPORT_DIR = "reports"
DEFAULT_LOG_DIR = "logs"
CUDA = False
FP16 = "O0"  # Optimization level for amp
DEBUG = False


def add_arguments(parser: ArgumentParser):
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--fp16", default="O0", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--debug", action="store_true")


def resolve_arguments(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    global FP16, CUDA, OVERWRITE_LOG, LOG_LEVEL_CONSOLE, DEBUG
    FP16 = args.fp16
    CUDA = args.cuda
    if args.debug:
        LOG_LEVEL_CONSOLE = "DEBUG"
        DEBUG = True
