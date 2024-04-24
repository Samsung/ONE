from circle_schema_generated import Model, ModelT
from train_info_viewer import TrainInfoViewer


class CirclePlusViewer:
    def __init__(self, fname: str):
        with open(fname, 'rb') as f:
            self.model = Model.GetRootAs(f.read(), 0)
            self.modelT = ModelT.InitFromObj(self.model)

    def show_details(self, data, indent: int):
        indentstr = ''
        for i in range(indent):
            indentstr += '\t'

        if isinstance(data, list):
            for idx, item in enumerate(data):
                print('{}{}: {}'.format(indentstr, idx, item))
                self.show_details(item, indent + 1)
        elif hasattr(data, '__dict__') and type(vars(data)) == dict:
            for key, value in vars(data).items():
                if isinstance(value, list):
                    print('{}{}: [....] (len:{})'.format(indentstr, key, len(value)))
                else:
                    print('{}{}: {}'.format(indentstr, key, value))
        else:
            print('{}{}'.format(indentstr, data))

    def show_metadata(self, key, value, indent):
        if key == 'metadata' and value != None:
            for item in value:
                if item.name == b'CIRCLE_TRAINING':
                    buffer = self.modelT.buffers[item.buffer]
                    tiv = TrainInfoViewer(buffer.data)
                    tiv.show(indent)

    def show(self, meta=True, details=True):
        for key, value in vars(self.modelT).items():
            print('{}:'.format(key))
            if details:
                self.show_details(value, 1)
            if details or meta:
                self.show_metadata(key, value, 2)
