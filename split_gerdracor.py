import argparse
import os
from os import path
import math


def split(input, train_size, dev_size):
    """
    Test set size will be whatever is left.
    """
    end_train = math.floor(len(input) * train_size)
    end_dev = end_train + math.ceil(len(input) * dev_size)
    return input[:end_train], input[end_train:end_dev], input[end_dev:]


def combine_files(conll_directory, out_name):
    all_file_names = list(sorted(
        [path.join(conll_directory, f) for f in os.listdir(conll_directory) if f.endswith(".Dirndl.conll")]
    ))
    splits = zip(split(all_file_names, 0.8, 0.1), ["train", "dev", "test"])
    for file_names, split_name in splits:
        print(f"{len(file_names)} files in {split_name}")
        dir_name, out_file_suffix = path.split(path.abspath(out_name))
        out_file_name = path.join(dir_name, split_name + out_file_suffix)
        out_file = open(out_file_name, "w")
        for file_name in file_names:
            file = open(file_name, "r")
            for line in file:
                out_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine conll files for training and evaluation")
    parser.add_argument("conll_directory", type=str, help="Directory with Dirndl conll files.")
    parser.add_argument("output_file_name", type=str, help="Path to the output file")
    parser.add_argument("--split", action="store_true")

    args = parser.parse_args()
    combine_files(args.conll_directory, args.output_file_name)
