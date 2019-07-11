import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("-o", "--output", default="averaged.pt")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoints[0], map_location="cpu")
    if "train" in checkpoint["state"]:
        del checkpoint["state"]["train"]

    params = checkpoint["state"]["model"]
    param_names = [name for name in params.keys() if isinstance(params[name], torch.Tensor)]

    with torch.no_grad():
        for filename in args.checkpoints[1:]:
            print(f"Loading from {filename}")

            new_state = torch.load(filename, map_location="cpu")["state"]["model"]
            for name in param_names:
                params[name] += new_state[name]

        for name in param_names:
            params[name].div_(len(args.checkpoints))

    torch.save(checkpoint, args.output)
