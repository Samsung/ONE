import json

from schema import circle_traininfo_generated as ctr_gen
'''
Wrappers of the auto generated classes in circle_traininfo_generated

Wrapper classes provides additional interfaces(e.g. initializer) to the auto generated classes.
'''

# Optimizers


class SGD(ctr_gen.SGDOptionsT):
    name = ['sgd', 'stocasticgradientdescent']


class Adam(ctr_gen.AdamOptionsT):
    name = ['adam']


# Loss


class SparseCategoricalCrossEntropy(ctr_gen.SparseCategoricalCrossentropyOptionsT):
    name = [
        'sparse categorical crossentropy', 'sparsecategoricalcrossentropy', 'sparsecce'
    ]


class CategoricalCrossEntropy(ctr_gen.CategoricalCrossentropyOptionsT):
    name = ['categorical crossentropy', 'categoricalcrossentropy', 'cce']


class MeanSqauaredError(ctr_gen.MeanSquaredErrorOptionsT):
    name = ['mean squared error', 'mse']


# TrainInfo


class TrainInfo(ctr_gen.ModelTrainingT):
    TRAINING_FILE_IDENTIFIER = b"CTR0"

    def __init__(self):
        super().__init__()

    @classmethod
    def from_buff(cls, buff):
        '''Create TrainInfo from buffer(byte array)'''
        tinfo = super().InitFromPackedBuf(bytearray(buff))
        tinfo.__class__ = TrainInfo

        # convert ModelTrainingT.optimizerOpt to wrapped class
        if tinfo.optimizer == ctr_gen.Optimizer.SGD:
            tinfo.optimizerOpt.__class__ = SGD
        elif tinfo.optimizer == ctr_gen.Optimizer.ADAM:
            tinfo.optimizerOpt.__class__ = Adam
        else:
            raise RuntimeError(f"Unknown optimizer {tinfo.optimizer}")

        # convert ModelTrainingT.lossfnOpt to wrapped class
        if tinfo.lossfn == ctr_gen.LossFn.SPARSE_CATEGORICAL_CROSSENTROPY:
            tinfo.lossfnOpt.__class__ = SparseCategoricalCrossEntropy
        elif tinfo.lossfn == ctr_gen.LossFn.CATEGORICAL_CROSSENTROPY:
            tinfo.lossfnOpt.__class__ = CategoricalCrossEntropy
        elif tinfo.lossfn == ctr_gen.LossFn.MEAN_SQUARED_ERROR:
            tinfo.lossfnOpt.__class__ = MeanSqauaredError
        else:
            raise RuntimeError(f"Unknown lossfn {tinfo.lossfn}")

        return tinfo

    def dump_as_json(self) -> str:
        '''Return JSON frommated string'''
        json_form = {}
        json_form["optimizer"] = {
            "type": self.optimizerOpt.name[0],
            "args": self.optimizerOpt.__dict__
        }
        json_form["loss"] = {
            "type": self.lossfnOpt.name[0],
            "args": self.lossfnOpt.__dict__,
        }
        reduction_str = {
            ctr_gen.LossReductionType.SumOverBatchSize: "SumOverBatchSize",
            ctr_gen.LossReductionType.Sum: "Sum",
        }
        json_form["loss"]["args"]["reduction"] = reduction_str[self.lossReductionType]
        json_form["batchSize"] = self.batchSize

        return json.dumps(json_form, indent=4)
