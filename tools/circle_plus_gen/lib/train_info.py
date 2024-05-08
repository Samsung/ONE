from schema.circle_traininfo_generated import *
from . import json_parser


# optimizers
class SGD(SGDOptionsT):
    name = ['sgd', 'stocasticgradientdescent']

    def __init__(self, learningRate=0.01):
        super().__init__()
        self.learningRate = learningRate


class Adam(AdamOptionsT):
    name = ['adam']

    def __init__(self, learningRate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-07):
        super().__init__()
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


# Loss
class SparseCategoricalCrossEntropy(SparseCategoricalCrossentropyOptionsT):
    name = [
        'sparse categorical crossentropy', 'sparsecategoricalcrossentropy', 'sparsecce'
    ]

    def __init__(self, fromLogits=False):
        super().__init__()
        self.fromLogits = fromLogits


class CategoricalCrossEntropy(CategoricalCrossentropyOptionsT):
    name = ['categorical crossentropy', 'categoricalcrossentropy', 'cce']

    def __init__(self, fromLogits=False):
        super().__init__()
        self.fromLogits = fromLogits


class MeanSqauaredError(MeanSquaredErrorOptionsT):
    name = ['mean squared error', 'mse']

    def __init__(self):
        super().__init__()


# TrainInfo
class TrainInfo(ModelTrainingT):

    TRAINING_FILE_IDENTIFIER = b"CTR0"

    def __init__(self):
        super().__init__()

    @classmethod
    def from_json(cls, json_obj: dict):
        tinfo = TrainInfo()

        opt, opt_type, opt_obj = json_parser.load_optimizer(json_obj["optimizer"])
        tinfo.optimizer = opt
        tinfo.optimizerOptType = opt_type
        tinfo.optimizerOpt = opt_obj

        lossfn, lossfn_type, lossfn_obj = json_parser.load_loss(json_obj["loss"])
        tinfo.lossfn = lossfn
        tinfo.lossfnOptType = lossfn_type
        tinfo.lossfnOpt = lossfn_obj

        tinfo.batchSize = json_obj["batchSize"]

        if "reduction" in json_obj["loss"].keys():
            tinfo.lossReductionType = json_parser.load_loss_rdt(
                json_obj["loss"]["reduction"])
        return tinfo

    @classmethod
    def from_buff(cls, buff):
        tinfo = super().InitFromPackedBuf(bytearray(buff))
        tinfo.__class__ = TrainInfo
        return tinfo

    def dump(self):
        ret = {}
        # dump optimizer
        opt_str = {Optimizer.SGD: SGD.name[0], Optimizer.ADAM: Adam.name[0]}
        ret["optimizer"] = {
            "type": opt_str[self.optimizer],
            "args": self.optimizerOpt.__dict__
        }

        # dump loss
        loss_str = {
            LossFn.SPARSE_CATEGORICAL_CROSSENTROPY: SparseCategoricalCrossEntropy.name[0],
            LossFn.CATEGORICAL_CROSSENTROPY: CategoricalCrossEntropy.name[0],
            LossFn.MEAN_SQUARED_ERROR: MeanSqauaredError.name[0]
        }
        ret["loss"] = {
            "type": loss_str[self.lossfn],
            "args": self.lossfnOpt.__dict__,
        }

        # dump reductions
        reduction_str = {
            LossReductionType.SumOverBatchSize: "SumOverBatchSize",
            LossReductionType.Sum: "Sum",
        }
        ret["loss"]["args"]["reduction"] = reduction_str[self.lossReductionType]

        # dump batchsize
        ret["batchSize"] = self.batchSize

        return ret

    def get_buff(self):
        builder = flatbuffers.Builder(0)
        builder.Finish(self.Pack(builder), self.TRAINING_FILE_IDENTIFIER)
        return builder.Output()
