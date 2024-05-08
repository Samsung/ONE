from . import train_info as tinfo
from schema.circle_traininfo_generated import *


def load_optimizer(opt_obj: dict):
    opt_type = opt_obj["type"]
    opt_args = opt_obj["args"]

    if opt_type.lower() in tinfo.SGD.name:
        return Optimizer.SGD, OptimizerOptions.SGDOptions, tinfo.SGD(**opt_args)

    elif opt_type.lower() in tinfo.Adam.name:
        return Optimizer.ADAM, OptimizerOptions.AdamOptions, tinfo.Adam(**opt_args)

    else:
        raise ValueError(f"not supported optmizer.type={opt_type}")


def load_loss_rdt(s: str):
    if s.lower() in ["sumoverbatchsize", "sum_over_batch_size"]:
        return tinfo.LossReductionType.SumOverBatchSize
    elif s.lower() in ["sum"]:
        return tinfo.LossReductionType.Sum
    else:
        raise ValueError(f"not supported loss.args.reduction={s}")


def load_loss(loss_obj: dict):
    loss_type = loss_obj["type"]
    loss_args = loss_obj["args"].copy()
    loss_args.pop("reduction")

    if loss_type.lower() in tinfo.SparseCategoricalCrossEntropy.name:
        return (LossFn.SPARSE_CATEGORICAL_CROSSENTROPY,
                LossFnOptions.SparseCategoricalCrossentropyOptions,
                tinfo.SparseCategoricalCrossentropyOptions(**loss_args))

    elif loss_type.lower() in tinfo.CategoricalCrossEntropy.name:
        return (LossFn.CATEGORICAL_CROSSENTROPY,
                LossFnOptions.CategoricalCrossentropyOptions,
                tinfo.CategoricalCrossEntropy(**loss_args))

    elif loss_type.lower() in tinfo.MeanSqauaredError.name:
        return (LossFn.MEAN_SQUARED_ERROR, LossFnOptions.MeanSquaredErrorOptions,
                tinfo.MeanSqauaredError())

    else:
        raise ValueError(f"not supported loss.type={loss_type}")
