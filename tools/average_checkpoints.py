import argparse

import torch


def convert_fp16(params, param_names):
    for name in param_names:
        tensor = params[name]
        if tensor.is_floating_point():
            params[name] = tensor.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("-o", "--output", default="averaged.pt")
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    print(f"Loading from {args.checkpoints[0]}")
    if args.cuda:
        checkpoint = torch.load(args.checkpoints[0])
    else:
        checkpoint = torch.load(args.checkpoints[0], map_location="cpu")
    if "train" in checkpoint["state"]:
        del checkpoint["state"]["train"]

    params = checkpoint["state"]["model"]
    param_names = [name for name in params.keys() if torch.is_tensor(params[name])]
    fp16 = [name for name in param_names if params[name].dtype is torch.half]
    if len(fp16) != 0:
        print("Converting checkpoints from fp16")
        convert_fp16(params, fp16)

    with torch.no_grad():
        for filename in args.checkpoints[1:]:
            print(f"Loading from {filename}")

            if args.cuda:
                new_state = torch.load(filename)["state"]["model"]
            else:
                new_state = torch.load(filename, map_location="cpu")["state"]["model"]
            convert_fp16(new_state, fp16)

            for name in param_names:
                params[name] += new_state[name]

        for name in param_names:
            params[name].div_(len(args.checkpoints))

    if fp16:
        print("Converting back to fp16")
        for name in fp16:
            params[name] = params[name].half()

    torch.save(checkpoint, args.output)


if __name__ == "__main__":
    main()
