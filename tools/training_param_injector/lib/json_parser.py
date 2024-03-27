import json
import typing

from . import train_info as tinfo


def load_optimizer(opt_obj: dict) -> tinfo.Optimizer:
    opt_type = opt_obj["type"]
    opt_args = opt_obj["args"]

    if (opt_type.lower() in tinfo.SGD.name):
        return tinfo.SGD(**opt_args)
    elif (opt_type.lower() in tinfo.Adam.name):
        return tinfo.Adam(**opt_args)
    else:
        raise ValueError(f"not supported optmizer.type={opt_type}")


def load_loss_rdt(s: str) -> tinfo.LossReduction:
    if (s.lower() in ["sumoverbatchsize", "sum_over_batch_size"]):
        return tinfo.LossReduction.SUM_OVER_BATCH_SIZE
    elif (s.lower() in ["sum"]):
        return tinfo.SUM
    else:
        raise ValueError(f"not supported loss.args.reduction={s}")


def load_loss(loss_obj: dict) -> tinfo.Loss:
    loss_type = loss_obj["type"]
    loss_args = loss_obj["args"]

    # update reduction string into corresponded enum
    if ("reduction" in loss_args.keys()):
        loss_args["reduction"] = load_loss_rdt(loss_args["reduction"])

    if (loss_type.lower() in tinfo.SparseCategoricalCrossentropy.name):
        return tinfo.SparseCategoricalCrossentropy(**loss_args)
    elif (loss_type.lower() in tinfo.CategoricalCrossentropy.name):
        return tinfo.CategoricalCrossentropy(**loss_args)
    elif (loss_type.lower() in tinfo.MeanSquaredError.name):
        return tinfo.MeanSquaredError(**loss_args)
    else:
        raise ValueError(f"not supported loss.type={loss_type}")


def load(json_file: str) -> tinfo.TrainingInfo:
    with open(json_file, 'rt') as f:
        json_obj = json.load(f)

    optimizer = load_optimizer(json_obj["optimizer"])
    loss = load_loss(json_obj["loss"])
    batch_size = json_obj["batch_size"]

    return tinfo.TrainingInfo(optimizer, loss, batch_size)
