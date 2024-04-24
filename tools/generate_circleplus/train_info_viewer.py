from circle_traininfo_generated import ModelTraining, ModelTrainingT


class TrainInfoViewer:
    def __init__(self, buf):
        self.tinfo = ModelTraining.GetRootAs(buf)
        self.tinfoT = ModelTrainingT.InitFromObj(self.tinfo)

    def show_details(self, data, indent):
        indentstr = ''
        for i in range(indent):
            indentstr += '\t'

        if hasattr(data, '__dict__') and type(vars(data)) == dict:
            for key, value in vars(data).items():
                if isinstance(value, list):
                    print('{}{}: [....] (len:{})'.format(indentstr, key, len(value)))
                else:
                    print('{}{}: {}'.format(indentstr, key, value))

    def show(self, indent: int = 0):
        indentstr = ''
        for i in range(indent):
            indentstr += '\t'

        print('{}[CIRCLE_TRAINING]'.format(indentstr))
        for key, value in vars(self.tinfoT).items():
            print('{}{}: {}'.format(indentstr, key, value))
            self.show_details(value, indent + 1)
