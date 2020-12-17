from os.path import dirname, realpath, join


class OpListParser():
    def __init__(self):
        self.file_name = "op_list.txt"
        self.op_list_file = join(dirname(realpath(__file__)), self.file_name)

    def parse(self):
        backend_op_list = {}
        with open(self.op_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                backend, _, op_list_str = line.partition(':')
                op_list = op_list_str.split(',')
                backend_op_list[backend] = op_list
        return backend_op_list
