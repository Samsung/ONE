from common_place import *
import caffe


def run_caffe(model_topology, model_weight, input_path, output_path=''):
    path = model_topology
    path_w = model_weight

    net = caffe.Net(path_w, path, caffe.TEST)
    # TODO get 'data' parameter more universal, blobs contain other names
    net.blobs['data'].data[...] = read_input(input_path)
    out = net.forward()
    all_names = [n for n in net._layer_names]
    out = out[all_names[-1]]
    save_result(output_path, out)
    print(out)


if __name__ == '__main__':
    args = regular_step()

    run_caffe(args.model[0], args.model[1], args.input, args.output_path)
