import tensorflow as tf

import argparse
import sys


def extract_subgraph(pb_path, output_node_names):
    with tf.Session() as sess:
        print("load graph")
        with tf.gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            graph_nodes = [n for n in graph_def.node]
            names = []
            for t in graph_nodes:
                names.append(t.name)
            print('nodes : ', names)

            if not output_node_names:
                print("You need to supply the name of a node to --output_node_names.")
                sys.exit(-1)

            return tf.compat.v1.graph_util.extract_sub_graph(graph_def,
                                                             output_node_names.split(","))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract subgraph from pb file')

    parser.add_argument("input_file", help="pb file to read")
    parser.add_argument("--output_node_names",
                        help="A list of strings specifying the destination node names.",
                        required=True)
    parser.add_argument("output_file", help="pb file to write")

    args = parser.parse_args()

    if not tf.gfile.Exists(args.input_file):
        print("Input graph file '" + args.input_file + "' does not exist!")
        sys.exit(-1)

    output_graph_def = extract_subgraph(args.input_file, args.output_node_names)

    with tf.gfile.GFile(args.output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
