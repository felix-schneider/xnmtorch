import argparse

import xnmtorch
from xnmtorch import logging
from xnmtorch.experiment import Experiment
from xnmtorch.logging import setup_logging
from xnmtorch.settings import add_arguments, resolve_arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval-only", action="store_true",
                        help="Run only the evaluation part of the experiment")
    parser.add_argument("experiment_file")
    parser.add_argument("spec_file", nargs="?")
    add_arguments(parser)

    args = parser.parse_args()
    resolve_arguments(args)

    setup_logging()
    logger = logging.getLogger("xnmtorch")
    logger.info(f"Running xnmtorch version {xnmtorch.__version__}")

    experiment = Experiment.load_experiment(args.experiment_file, args.spec_file,
                                            for_training=not args.eval_only)

    experiment()


if __name__ == '__main__':
    main()
