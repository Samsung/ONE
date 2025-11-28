#!/usr/bin/env python3

import numpy as np
import random
import sys
from onert import infer


def main(nnpackage_path, backends="cpu"):
    # Create session and load nnpackage
    session = infer.session(nnpackage_path, backends)

    # Prepare input. Here we just allocate dummy input arrays.
    input_infos = session.get_inputs_tensorinfo()

    # Call infer() 10 times
    for i in range(10):
        dummy_inputs = []
        for info in input_infos:
            # Retrieve the dimensions list from tensorinfo property.
            dims = list(info.dims)
            # Replace -1 with a random value between 1 and 10
            dims = [random.randint(1, 10) if d == -1 else d for d in dims]
            # Build the shape tuple from tensorinfo dimensions.
            shape = tuple(dims[:info.rank])
            # Create a dummy numpy array filled with uniform random values in [0,1).
            dummy_inputs.append(
                np.random.uniform(low=0.0, high=1.0, size=shape).astype(info.dtype))

        outputs = session.infer(dummy_inputs)
        print(f"Inference run {i+1}/10 completed.")

    print(f"nnpackage {nnpackage_path.split('/')[-1]} runs successfully.")
    return


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
