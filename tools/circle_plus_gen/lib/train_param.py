import flatbuffers
import json

from lib.utils import *
from lib.json_parser import *
from schema import circle_traininfo_generated as ctr_gen


class TrainParam():
    '''Wrapper class of circle_traninfo_generated.ModelTrainingT'''
    TRAINING_PARAM_IDENTIFIER = b"CTR0"

    def __init__(self):
        self.train_param = ctr_gen.ModelTrainingT()

    @classmethod
    def from_buff(cls, buff):
        '''Create TrainInfo from packed(serialized) buffer'''
        new_tparam = cls()
        new_tparam.train_param = ctr_gen.ModelTrainingT.InitFromPackedBuf(bytearray(buff))
        return new_tparam

    def to_buff(self):
        '''Serialize train_param and return its buffer'''
        builder = flatbuffers.Builder(0)
        builder.Finish(self.train_param.Pack(builder), self.TRAINING_PARAM_IDENTIFIER)
        return builder.Output()

    @classmethod
    def from_json(cls, json_file: str, num_op: int):
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

        # load fine_tuning
        fine_tuning = 0  # if not given, full training is default
        if "fineTuning" in json_obj:
            fine_tuning = json_obj["fineTuning"]
        try:
            tparam.trainableOps = load_fine_tuning(fine_tuning, num_op)
        except ValueError as e:
            print(e)
            raise (f"failed to parse \'fineTuning\'")

        new_tparam = cls()
        new_tparam.train_param = tparam
        return new_tparam

    def dump_as_json(self, num_op: int) -> str:
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

        ft = []
        if tparam.trainableOps != None:
            ft = list(tparam.trainableOps)
        num_ft = len(ft)

        if num_ft == 0:
            json_form["fineTuning"] = -1

        elif ft == list(range(0, num_op)):
            json_form["fineTuning"] = 0

        elif num_ft < num_op and list(range(num_op - num_ft, num_op)) == ft:
            json_form["fineTuning"] = num_ft

        else:
            raise ValueError(f"fail to dump fineTuning{ft}")

        return json.dumps(json_form, indent=4)
