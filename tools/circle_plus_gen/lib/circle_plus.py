import typing

from schema import circle_schema_generated as cir_gen
from lib.train_param import TrainParam


class CirclePlus():
    ''' Wrapper class of circle_schema_generated.ModelT'''
    TINFO_META_TAG = "CIRCLE_TRAINING"

    def __init__(self):
        self.model: cir_gen.ModelT = cir_gen.ModelT()

    @classmethod
    def from_file(cls, circle_file: str):
        '''Create CirclePlus based on circle file'''
        new_circle_plus = cls()
        with open(circle_file, 'rb') as f:
            new_circle_plus.model = cir_gen.ModelT.InitFromPackedBuf(f.read())
        return new_circle_plus

    def get_train_param(self) -> typing.Union[TrainParam, None]:
        '''Return TrainInfo, if it exists'''
        metadata = self.model.metadata
        buffers = self.model.buffers

        if metadata == None:
            return None

        for m in metadata:
            if m.name.decode("utf-8") == self.TINFO_META_TAG:
                buff: cir_gen.BufferT = buffers[m.buffer]
                tparam: TrainParam = TrainParam.from_buff(buff.data)
                return tparam

        return None
