import flatbuffers
import typing

from schema import circle_schema_generated as cir_gen
from lib.train_param import TrainParam


class CirclePlus():
    ''' Wrapper class of circle_schema_generated.ModelT'''
    TINFO_META_TAG = "CIRCLE_TRAINING"
    CIRCLE_IDENTIFIER = b"CIR0"

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

    def _add_metadata(self, meta_name, meta_buf):
        buffer_obj = cir_gen.BufferT()
        buffer_obj.data = meta_buf

        # If there are train_param, Replace it
        if self.get_train_param() is not None:
            for m in self.model.metadata:
                if m.name.decode("utf-8") == self.TINFO_META_TAG:
                    self.model.buffers[m.buffer] = buffer_obj

        # There are no train_param, So add a new buffer and metadata
        else:
            if self.model.metadata == None:
                self.model.metadata = []

            if self.model.buffers == None:
                self.model.buffers = []

            self.model.buffers.append(buffer_obj)

            metadata_obj = cir_gen.MetadataT()
            metadata_obj.name = meta_name
            metadata_obj.buffer = len(self.model.buffers) - 1
            self.model.metadata.append(metadata_obj)

    def set_train_param(self, train_param: TrainParam):
        '''Add train_param to the model's metadata field'''
        tparam_buff = train_param.to_buff()
        self._add_metadata(self.TINFO_META_TAG, tparam_buff)

    def export(self, circle_file: str):
        '''Export model to the circle file'''
        builder = flatbuffers.Builder(0)
        builder.Finish(self.model.Pack(builder), self.CIRCLE_IDENTIFIER)

        with open(circle_file, 'wb') as f:
            f.write(builder.Output())
