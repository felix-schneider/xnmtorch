import argparse
import logging

import h5py
import numpy as np
from tqdm import tqdm

import xnmtorch
from xnmtorch.data.vocab import Vocab
from xnmtorch.logging import setup_logging

CHUNK_SIZE = 10000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("source_file")
    parser.add_argument("target_file")
    parser.add_argument("vocab_file")
    parser.add_argument("output_file", default="train.h5")
    parser.add_argument("-m", "--max-len", default=150)

    args = parser.parse_args()
    setup_logging()

    logger = logging.getLogger()
    logger.info(f"Running xnmtorch version {xnmtorch.__version__}")

    def filter_example(example):
        return len(example.src) <= args.max_len and len(example.trg) <= args.max_len

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

    dt_vlen = h5py.special_dtype(vlen=dtype)
    # dt_arr = np.dtype((dt_vlen, (args.max_len,)))
    dt = np.dtype([("src", dt_vlen), ("trg", dt_vlen)])

    with open(args.source_file) as src_file, open(args.target_file) as trg_file:
        logger.info("Loading")
        examples = []
        src_lengths = []
        trg_lengths = []
        for i, (src_line, trg_line) in enumerate(tqdm(zip(src_file, trg_file))):
            src_example = np.array([vocab.stoi[w] for w in src_line.rstrip("\n").split()], dtype=dtype)
            trg_example = np.array([vocab.stoi[w] for w in trg_line.rstrip("\n").split()], dtype=dtype)
            if len(src_example) > args.max_len or len(trg_example) > args.max_len:
                continue
            examples.append((src_example, trg_example))
            src_lengths.append(len(src_example))
            trg_lengths.append(len(trg_example))

    logger.info("Sorting")
    ds_len = len(examples)
    indices = sorted(range(ds_len), key=lambda ii: (trg_lengths[ii], src_lengths[ii]))
    examples = [examples[x] for x in indices]
    src_lengths = [src_lengths[x] for x in indices]
    trg_lengths = [trg_lengths[x] for x in indices]
    logger.info("Sorted, writing output")
    
    with h5py.File(args.output_file, "w") as out_file, tqdm(total=ds_len) as pbar:
        src_len_ds = out_file.create_dataset("src_len", (ds_len,), dtype='uint8')
        src_len_ds[:] = src_lengths
        del src_lengths

        trg_len_ds = out_file.create_dataset("trg_len", (ds_len,), dtype='uint8')
        trg_len_ds[:] = trg_lengths
        del trg_lengths
        
        ds = out_file.create_dataset("examples", (ds_len,), dtype=dt)
        
        offset = 0
        chunk = np.empty((CHUNK_SIZE,), dtype=dt)

        for i, example in enumerate(examples):
            chunk[i % CHUNK_SIZE] = example
            if (i + 1) % CHUNK_SIZE == 0:
                ds[offset:i+1] = chunk
                offset = i+1
                pbar.update(CHUNK_SIZE)

        if ds_len % CHUNK_SIZE != 0:
            ds[offset:] = chunk[:ds_len % CHUNK_SIZE]
            pbar.update(ds_len % CHUNK_SIZE)

    logger.info("Done")
