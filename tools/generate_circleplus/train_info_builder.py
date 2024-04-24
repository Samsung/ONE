import flatbuffers
import circle_traininfo_generated as ti
from circle_traininfo_generated import ModelTrainingT


class GenerateOptions:
    def __init__(self, obj):
        self.optionstr = {}

        attrs = self._get_attrs(obj)
        for attr in attrs:
            if attr == 'NONE':
                continue
            key = attr.replace('Options', '')
            idx = getattr(obj, attr)
            optionT = getattr(ti, attr + 'T', None)
            if optionT:
                self.optionstr[key] = [idx, optionT()]
            else:
                self.optionstr[key] = idx

    def _get_attrs(self, obj):
        return [
            attr for attr in dir(obj)
            if not callable(getattr(obj, attr)) and not attr.startswith('__')
        ]

    def builder(self, argtype, args):
        optionT = self.optionstr[argtype][1]
        for option, _ in vars(optionT).items():
            value = getattr(args, option, None)
            if value:
                typev = type(getattr(optionT, option))(value)
                setattr(optionT, option, typev)
        return optionT


class TrainInfoLoss:
    # NOTE
    # Use LossFnOptions instead of LossFn to access LossFnOptions easily.
    # The type string between LossFn and LossFnOption does not match.
    _options = GenerateOptions(ti.LossFnOptions)

    @classmethod
    def options(cls):
        return cls._options.optionstr

    @classmethod
    def types(cls):
        return list(cls.options().keys())

    @classmethod
    def default(cls):
        return cls.types()[0]

    @classmethod
    def lossfn(cls, type):
        # NOTE
        # LossFn enum is 1 less than LossFnOptions enum because of NONE.
        return cls.lossfnOptType(type) - 1

    @classmethod
    def lossfnOptType(cls, type):
        return cls.options()[type][0]

    @classmethod
    def lossfnOpt(cls, type, args):
        return cls._options.builder(type, args)


class TrainInfoLossReduction:
    _options = GenerateOptions(ti.LossReductionType)

    @classmethod
    def options(cls):
        return cls._options.optionstr

    @classmethod
    def types(cls):
        return list(cls.options().keys())

    @classmethod
    def default(cls):
        return cls.types()[0]

    @classmethod
    def lossReductionType(cls, type):
        return cls.options()[type]


class TrainInfoOptimizer:
    # NOTE
    # Use OptimizerOptions instead of Optimizer to access OptimizerOptions easily.
    # The type string between Optimizer and OptimizerOptions does not match.
    _options = GenerateOptions(ti.OptimizerOptions)

    @classmethod
    def options(cls):
        return cls._options.optionstr

    @classmethod
    def types(cls):
        return list(cls.options().keys())

    @classmethod
    def default(cls):
        return cls.types()[0]

    @classmethod
    def arguments(cls, opt):
        return vars(cls.options()[opt][1])

    @classmethod
    def optimizer(cls, type):
        return cls.optimizerOptType(type) - 1

    @classmethod
    def optimizerOptType(cls, type):
        return cls.options()[type][0]

    @classmethod
    def optimizerOpt(cls, type, args):
        return cls._options.builder(type, args)


class TrainInfoBuilder:
    TRAINING_FILE_VERSION = 1
    TRAINING_FILE_IDENTIFIER = b'CTR0'

    def __init__(self, lossfn: int, lossfnOptType: int, lossfnOpt, optimizer: int,
                 optimizerOptType: int, optimizerOpt, batchSize: int,
                 lossReductionType: int):
        self.tinfo = ModelTrainingT()
        self.tinfo.version = self.TRAINING_FILE_VERSION
        self.tinfo.lossfn = lossfn
        self.tinfo.lossfnOptType = lossfnOptType
        self.tinfo.lossfnOpt = lossfnOpt
        self.tinfo.optimizer = optimizer
        self.tinfo.optimizerOptType = optimizerOptType
        self.tinfo.optimizerOpt = optimizerOpt
        self.tinfo.batchSize = batchSize
        self.tinfo.lossReductionType = lossReductionType

    def get(self):
        builder = flatbuffers.Builder(0)
        builder.Finish(self.tinfo.Pack(builder), self.TRAINING_FILE_IDENTIFIER)
        return builder.Output()
