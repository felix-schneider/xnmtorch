import argparse
import os

from apex import amp

import xnmtorch
from xnmtorch import logging
from xnmtorch.data.datasets import TranslationDataset
from xnmtorch.eval.eval_tasks import DecodingEvalTask
from xnmtorch.eval.metrics import BLEU
from xnmtorch.eval.search_strategies import BeamSearch
from xnmtorch.experiment import Experiment, ExpGlobal
from xnmtorch.logging import setup_logging
from xnmtorch.settings import add_arguments, resolve_arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_file")
    parser.add_argument("data_path")
    parser.add_argument("output_path", default="output.hyp")
    parser.add_argument("-e", "--ext", "--extensions", nargs=2, default=[".src", ".ref"])
    parser.add_argument("-a", "--alpha", type=float, default=0, help="Length penalty")
    parser.add_argument("-b", "--beam-size", type=int, default=5)
    parser.add_argument("-n", "--batch_size", type=int, default=32)
    parser.add_argument("-s", "--sort-within-batch", action="store_true")
    parser.add_argument("-m", "--multiple", type=int, default=1)
    parser.add_argument("-p", "--print-output", action="store_true")
    parser.add_argument("-l", "--lowercase", action="store_true")
    parser.add_argument("--max-len", type=int, default=100)
    add_arguments(parser)

    args = parser.parse_args()
    resolve_arguments(args)

    setup_logging()
    logger = logging.getLogger("xnmtorch")
    logger.info(f"Running xnmtorch version {xnmtorch.__version__}")

    model = Experiment.load_model(args.experiment_file)

    eval_task = DecodingEvalTask(
        TranslationDataset(
            path=args.data_path,
            batch_size=args.batch_size,
            extensions=args.ext,
            src_vocab=model.src_vocab,
            trg_vocab=model.trg_vocab,
            level=model.level,
            batch_by_words=False,
            sort_within_batch=args.sort_within_batch,
            batch_first=model.batch_first,
            multiple=args.multiple
        ),
        metrics=BLEU(lowercase=args.lowercase),
        search_strategy=BeamSearch(args.beam_size, length_penalty=args.alpha, max_len_b=args.max_len),
        model=model,
        name="test",
        report_dir="reports",
        result_path=args.output_path,
        print_output=args.print_output
    )
    experiment = Experiment(ExpGlobal(), model, evaluate=eval_task)

    experiment()


if __name__ == "__main__":
    main()
