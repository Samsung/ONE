import json

from lib.utils import *
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
