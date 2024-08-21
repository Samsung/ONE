from onert import infer
import sys


def main(nnpackage_path, backends="cpu"):
    # Create session and load nnpackage
    # The default value of backends is "cpu".
    session = infer.session(nnpackage_path, backends)

    # Prepare input. Here we just allocate dummy input arrays.
    input_size = session.input_size()
    session.set_inputs(input_size)

    outputs = session.inference()

    print(f"nnpackage {nnpackage_path.split('/')[-1]} runs successfully.")
    return


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
