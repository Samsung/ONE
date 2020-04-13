## pb_info.py
- prints information inside `pb` file.
- how to run:
    - `./tools/pbfile_tool/pb_info.py pbfile_path "conv"`
        - first arg: pb file
        - second arg: substring of operation. Only operations that has "conv" substring as its type will be printed. (case-insensitive)
    - `./tools/pbfile_tool/pb_info.py pbfile_path "*"`
        - pass "*" as the second param to print all operations
    - `./tools/pbfile_tool/pb_info.py pbfile_path "*" --summary`
        - prints the list of operations and their counts
    - `./tools/pbfile_tool/pb_info.py pbfile_path "*" --summary --name_prefix=Model/rnn`
        - prints the summary of operations of which names start `Model/rnn`

## convert_ckpt_to_pb.py
- convert checkpoint file to pb file and _freeze_ the `pb` file.
- how to run:
    - `$ PYTHONPATH=tools/tensorflow_model_freezer/ python convert_ckpt_to_pb.py checkpoint_dir checkpoint_file_name`

## extract_subgraph.py
- extract a subgraph from `pb` file and save as the subgraph `pb` file.
- how to run:
    - `python extract_subgraph.py input_file output_file --output_node_names output_node_names`
