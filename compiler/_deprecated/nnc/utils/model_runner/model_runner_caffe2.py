from common_place import *

from caffe2.python import workspace


def run_caffe2(init_net, predict_net, input_path, output_path=''):
    x = read_input(input_path)
    with open(init_net, 'rb') as f:
        init_net = f.read()

    with open(predict_net, 'rb') as f:
        predict_net = f.read()
    p = workspace.Predictor(init_net, predict_net)
    # TODO get 'data' parameter more universal, blobs contain other names
    results = p.run({'data': x})
    print(results)
    save_result(output_path, results)


if __name__ == '__main__':
    args = regular_step()

    run_caffe2(args.model[0], args.model[1], args.input, args.output_path)
