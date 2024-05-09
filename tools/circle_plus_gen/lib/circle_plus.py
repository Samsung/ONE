import typing

from schema import circle_schema_generated as cir_gen
from lib.train_info import TrainInfo


class CirclePlus(cir_gen.ModelT):
    '''
    Wrapper of auto generated circle_schema_generated.ModelT
    '''
    TINFO_META_TAG = "CIRCLE_TRAINING"

    def __init__(self):
        super().__init__()

    @classmethod
    def from_file(cls, circle_file: str):
        '''Create CirclePlus based on circle file'''
        with open(circle_file, 'rb') as f:
            circle = super().InitFromPackedBuf(f.read())
            circle.__class__ = CirclePlus
            return circle

    def get_train_info(self) -> typing.Union[TrainInfo, None]:
        '''Return TrainInfo, if it exists'''
        if not self.metadata:
            return None

        for meta in self.metadata:
            if meta.name.decode("utf-8") == self.TINFO_META_TAG:
                buff: cir_gen.BufferT = self.buffers[meta.buffer]
                tinfo: TrainInfo = TrainInfo.from_buff(buff.data)
                return tinfo

        return None
