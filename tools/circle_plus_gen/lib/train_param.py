import json

from lib.utils import *
from lib.json_parser import *
from schema import circle_traininfo_generated as ctr_gen


class TrainParam():
    '''Wrapper class of circle_traninfo_generated.ModelTrainingT'''

    def __init__(self):
        self.train_param = ctr_gen.ModelTrainingT()

    @classmethod
    def from_buff(cls, buff):
        '''Create TrainInfo from packed(serialized) buffer'''
        new_tparam = cls()
        new_tparam.train_param = ctr_gen.ModelTrainingT.InitFromPackedBuf(bytearray(buff))
        return new_tparam

    @classmethod
    def from_json(cls, json_file: str):
        '''Create TrainInfo from json file'''
        with open(json_file, 'rt') as f:
            json_obj = json.load(f)

        tparam = ctr_gen.ModelTrainingT()

        # load optimzier
        optim, optim_opt_type, optim_opt = load_optimizer(json_obj["optimizer"])
        tparam.optimizer = optim
        tparam.optimizerOptType = optim_opt_type
        tparam.optimizerOpt = optim_opt

        # load lossfn
        lossfn, lossfn_opt_type, lossfn_opt = load_lossfn(json_obj["loss"])
        tparam.lossfn = lossfn
        tparam.lossfnOptType = lossfn_opt_type
        tparam.lossfnOpt = lossfn_opt

        tparam.batchSize = json_obj["batchSize"]

        # load lossReductionType
        if "reduction" in json_obj["loss"].keys():
            tparam.lossReductionType = load_loss_reduction(json_obj["loss"]["reduction"])

        new_tparam = cls()
        new_tparam.train_param = tparam
        return new_tparam

    def dump_as_json(self) -> str:
        '''Return JSON formmated string'''
        tparam = self.train_param
        name_opt = OptimizerNamer()
        name_loss = LossNamer()
        name_rdt = LossReductionNamer()

        json_form = {}
        json_form["optimizer"] = {
            "type": name_opt(type(tparam.optimizerOpt))[0],
            "args": tparam.optimizerOpt.__dict__
        }
        json_form["loss"] = {
            "type": name_loss(type(tparam.lossfnOpt))[0],
            "args": tparam.lossfnOpt.__dict__,
        }
        json_form["loss"]["args"]["reduction"] = name_rdt(tparam.lossReductionType)[0]
        json_form["batchSize"] = tparam.batchSize

        return json.dumps(json_form, indent=4)
