import re
from typing import Tuple, List, Union

from lib import utils
from schema import circle_traininfo_generated as ctr_gen


def to_camel_case(string: str):
    if not '_' in string:
        # string is already camel case
        return string

    captialized_str: str = "".join(x.capitalize() for x in string.lower().split("_"))
    return string[0].lower() + captialized_str[1:]


def _generate_optimizer(
        opt_type: utils.OPTIM_OPTIONS_T, args: dict
) -> Tuple[ctr_gen.Optimizer, ctr_gen.OptimizerOptions, utils.OPTIM_OPTIONS_T]:

    options_t_str: str = opt_type.__name__  # e.g. SGDOptionsT
    options_str: str = options_t_str[:-1]  # e.g. SGDOptions
    optimizer_camel: str = options_str.replace("Options", "")  # e.g. SGD
    optimizer_str: str = re.sub(r'(?<=[a-z])(?=[A-Z])', '_',
                                optimizer_camel).upper()  # e.g. SGD

    optimizer = getattr(ctr_gen.Optimizer, optimizer_str)
    optimizer_opt_type = getattr(ctr_gen.OptimizerOptions, options_str)

    # set attributes for *OptionsT
    # e.g. SGDOptionsT.learningRate = 0.001
    optimizer_opt = opt_type()
    for (key, value) in args.items():
        key = to_camel_case(key)
        setattr(optimizer_opt, key, value)

    return optimizer, optimizer_opt_type, optimizer_opt


def load_optimizer(
        opt_obj: dict
) -> Tuple[ctr_gen.Optimizer, ctr_gen.OptimizerOptions, utils.OPTIM_OPTIONS_T]:
    ''' 
    Return objects for circle_traininfo_generated.ModelTrainingT.[optimizer, optimizerOptType, OptimizerOpt]
    
    An example of given arguments and return values : 
    
        opt_obj : {
          "type" : "sgd", 
          "args" : {"learningRate = 0.1"}
        }
        return : (Optimizer.SGD, OptimizerOptions.SGDOptions, object of SGDOptionsT)
    '''
    opt_type = opt_obj["type"]
    opt_args = opt_obj["args"]

    names_of = utils.OptimizerNamer()
    supported_opt = [ctr_gen.SGDOptionsT, ctr_gen.AdamOptionsT]

    # find type(e.g. SGDOptionsT) from opt_type("sgd")
    type = None
    for t in supported_opt:
        if opt_type.lower() in names_of(t):
            type = t
            break
    if type == None:
        raise ValueError(f"not supported optimizer.type={opt_type}")

    return _generate_optimizer(type, opt_args)


def _generate_lossfn(
        lossfn_type: utils.LOSSFN_OPTIONS_T, args: dict
) -> Tuple[ctr_gen.LossFn, ctr_gen.LossFnOptions, utils.LOSSFN_OPTIONS_T]:

    options_t_str: str = lossfn_type.__name__  # e.g. CategoricalCrossentropyOptionsT
    options_str: str = options_t_str[:-1]  # e.g. CategoricalCrossentropyOptions
    lossfn_camel: str = options_str.replace("Options", "")  # e.g. CategoricalCrossentropy
    lossfn_str: str = re.sub(r'(?<=[a-z])(?=[A-Z])', '_',
                             lossfn_camel).upper()  # e.g. CATEGORICAL_CROSSENTROPY

    lossfn = getattr(ctr_gen.LossFn, lossfn_str)
    lossfn_opt_type = getattr(ctr_gen.LossFnOptions, options_str)

    # set attributes for *OptionsT
    # e.g. CategoricalCrossentropyOptionsT.fromLogits = True
    lossfn_opt = lossfn_type()
    for (key, value) in args.items():
        key = to_camel_case(key)
        setattr(lossfn_opt, key, value)

    return lossfn, lossfn_opt_type, lossfn_opt


def load_lossfn(loss_obj: dict):
    '''
    Return objects for circle_traininfo_generated.ModelTrainingT.[lossfn, lossfnOptType, lossfnOpt]

    An example of given arguments and return values :
    
        loss_obj : {
          "type" : "categorical crossentropy", 
          "args" : {"fromLogits = True"}
        }
        return : (LossFn.CATEGORICAL_CROSSENTROPY, 
                  LossFnOptions.CategoricalCrossentropyOptions,
                  object of CategoricalCrossentropyOptionsT)
    '''
    loss_type = loss_obj["type"]
    loss_args = loss_obj["args"].copy()
    loss_args.pop("reduction")

    names_of = utils.LossNamer()
    # yapf:disable
    supported_loss = [
        ctr_gen.SparseCategoricalCrossentropyOptionsT,
        ctr_gen.CategoricalCrossentropyOptionsT,
        ctr_gen.MeanSquaredErrorOptionsT
    ]
    # yapf: enable

    # find type(e.g. MeanSquaredErrorOptionsT) from loss_type(e.g. "mean squared error")
    type = None
    for t in supported_loss:
        if loss_type.lower() in names_of(t):
            type = t
            break
    if type == None:
        raise ValueError(f"not supported loss.type={loss_type}")

    return _generate_lossfn(type, loss_args)


def load_loss_reduction(s: str):
    ''' Return LossReductionType enum which is correspond to given string
    '''
    names_of = utils.LossReductionNamer()
    supported_rdt = [
        ctr_gen.LossReductionType.SumOverBatchSize, ctr_gen.LossReductionType.Sum
    ]

    type = None
    for t in supported_rdt:
        if s.lower() in names_of(t):
            type = t
            break
    if type == None:
        raise ValueError(f"not supported loss.args.reduction={s}")

    return type
