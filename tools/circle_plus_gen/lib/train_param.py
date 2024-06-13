from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.circle_plus import CirclePlus

import flatbuffers
import json
import numpy as np

from lib.utils import *
from lib.json_parser import *
from schema import circle_traininfo_generated as ctr_gen


class TrainParam():
    '''Wrapper class of circle_traninfo_generated.ModelTrainingT'''
    TRAINING_PARAM_IDENTIFIER = b"CTR0"

    def __init__(self, circle_model: CirclePlus):
        self.train_param = ctr_gen.ModelTrainingT()
        self.circle_model = circle_model

    @classmethod
    def from_buff(cls, circle_model: CirclePlus, buff):
        '''Create TrainInfo from packed(serialized) buffer'''
        new_tparam = cls(circle_model)
        new_tparam.train_param = ctr_gen.ModelTrainingT.InitFromPackedBuf(bytearray(buff))
        return new_tparam

    def to_buff(self):
        '''Serialize train_param and return its buffer'''
        builder = flatbuffers.Builder(0)
        builder.Finish(self.train_param.Pack(builder), self.TRAINING_PARAM_IDENTIFIER)
        return builder.Output()

    def set_trainable(self, trainable: List[int]):
        self.train_param.trainableOps = trainable

    def get_trainable(self) -> List[int]:
        return self.train_param.trainableOps

    @classmethod
    def from_json(cls, circle_model: CirclePlus, json_file: str):
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

        # load trainable
        train = "all"  # default, train all operators
        if "train" in json_obj.keys():
            train = json_obj["train"]

        try:
            tparam.trainableOps = load_trainable(train,
                                                 circle_model.get_number_of_operators())
        except ValueError as e:
            print(e)
            raise (f"failed to parse \'train\'")

        new_tparam = cls(circle_model)
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

        total = self.circle_model.get_number_of_operators()
        trainable = len(tparam.trainableOps)

        if trainable == 0:
            json_form["trainable"] = "none"
        elif trainable == total and (np.arange(0,
                                               trainable) == tparam.trainableOps).all():
            json_form["trainable"] = "all"
        elif trainable < total and (np.arange(total - trainable,
                                              total) == tparam.trainableOps).all():
            json_form["trainable"] = "last" + str(trainable)
        else:
            raise ValueError(f"trainable Ops has unexpected values {tparam.trainableOps}")

        return json.dumps(json_form, indent=4)
