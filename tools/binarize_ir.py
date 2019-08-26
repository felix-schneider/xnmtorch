import argparse
import logging
import subprocess

import h5py
import numpy as np
from tqdm import tqdm

import xnmtorch
from xnmtorch.data.vocab import Vocab
from xnmtorch.logging import setup_logging


CHUNK_SIZE = 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("questions_file")
    parser.add_argument("paragraphs_file")
    parser.add_argument("vocab_file")
    parser.add_argument("output_file")

    args = parser.parse_args()
    setup_logging()

    logger = logging.getLogger()
    logger.info(f"Running xnmtorch version {xnmtorch.__version__}")

    try:
        vocab = Vocab(args.vocab_file)
    except ValueError:
        vocab = Vocab(args.vocab_file, sentence_piece=True)

    if len(vocab) < 2 ** 8:
        dtype = 'uint8'
    elif len(vocab) < 2 ** 16:
        dtype = 'uint16'
    elif len(vocab) < 2 ** 32:
        dtype = 'uint32'
    else:
        dtype = 'uint64'

    ds_len = int(subprocess.check_output(["wc", "-l", args.questions_file]).split()[0])
    n_paragraphs = int(subprocess.check_output(["wc", "-l", args.paragraphs_file]).split()[0])
    assert n_paragraphs % ds_len == 0
    ratio = n_paragraphs // ds_len

    dt_vlen = h5py.special_dtype(vlen=dtype)
    dt = np.dtype([("question", dt_vlen), ("paragraphs", dt_vlen), ("length", "uint16")])

    with open(args.questions_file) as questions, open(args.paragraphs_file) as paragraphs, \
        h5py.File(args.output_file, "w") as out_file, tqdm(total=ds_len) as pbar:
        ds = out_file.create_dataset("examples", (ds_len,), dtype=dt)
        ds.attrs["ratio"] = ratio

        offset = 0
        chunk = np.empty((CHUNK_SIZE,), dtype=dt)

        for i, question in enumerate(questions):
            question = np.array([vocab.stoi[w] for w in question.rstrip("\n").split()], dtype=dtype)
            ps = [[vocab.stoi[w] for w in paragraphs.readline().rstrip("\n").split()]
                  for _ in range(ratio)]
            max_len = max(len(p) for p in ps)
            p_array = np.full((ratio, max_len), vocab.pad_index)
            for j, p in enumerate(ps):
                p_array[j, :len(p)] = p
            example = (question, p_array.flatten(), max_len)
            chunk[i % CHUNK_SIZE] = example
            if (i + 1) % CHUNK_SIZE == 0:
                ds[offset:i+1] = chunk
                offset = i+1
                pbar.update(CHUNK_SIZE)

        if ds_len % CHUNK_SIZE != 0:
            ds[offset:] = chunk[:ds_len % CHUNK_SIZE]
            pbar.update(ds_len % CHUNK_SIZE)

    logger.info("Done")
