import argparse
import numpy as np
from numpy import NDArray


def count_bits_in_row(list: NDArray[np.uint64]):
    # Use numapy's binary_repr to get binary strings, then count '1's
    one_count = sum(np.binary_repr(x, width=64).count('1') for x in list)
    zero_count = list.size * 64 - one_count
    return zero_count, one_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count the number of filled cells in a row of np.uint64 integers.")
    parser.add_argument("list", type=list, help="Comma-separated list of np.uint64 integers.")
    args = parser.parse_args()

    print(count_bits_in_row(np.array(args.list, dtype=np.uint64)))
