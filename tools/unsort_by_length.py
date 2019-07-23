import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file")
    parser.add_argument("sorted_file")
    parser.add_argument("output_file")

    args = parser.parse_args()

    with open(args.source_file) as source_file:
        lengths = [len(line.split()) for line in source_file]

    indices = sorted(range(len(lengths)), key=lengths.__getitem__)
    inverted_indices = sorted(range(len(indices)), key=indices.__getitem__)

    with open(args.sorted_file) as sorted_file:
        sorted_lines = sorted_file.readlines()

    with open(args.output_file, "w") as output_file:
        output_file.writelines(sorted_lines[i] for i in inverted_indices)


if __name__ == "__main__":
    main()
