import flatbuffers
import circle_schema_generated as circle
import circle_traininfo_generated as circleti
import train_info_viewer as tiviewer


class CirclePlusBuilder:
    '''
  CirclePlus Model Builder
  '''
    CIRCLE_FILE_IDENTIFIER = b'CIR0'

    def __init__(self, fname: str):
        with open(fname, 'rb') as f:
            self.model = circle.Model.GetRootAs(f.read(), 0)
            self.modelT = circle.ModelT.InitFromObj(self.model)

    def injectMetaData(self, mName, mData):
        buffer = circle.BufferT()
        buffer.data = mData

        if self.modelT.metadata is None:
            self.modelT.metadata = []

        found = False
        for meta in self.modelT.metadata:
            if meta.name == mName.encode('utf-8'):
                self.modelT.buffers[meta.buffer] = buffer
                found = True

        if found == False:
            metadata = circle.MetadataT()
            metadata.name = mName.encode('utf-8')
            metadata.buffer = len(self.modelT.buffers)
            self.modelT.metadata.append(metadata)
            self.modelT.buffers.append(buffer)

    def export(self, fname: str):
        if self.modelT.metadata:
            for meta in self.modelT.metadata:
                if meta.name == b'CIRCLE_TRAINING':
                    buf_idx = meta.buffer
            buffer = self.modelT.buffers[buf_idx]
            tiv = tiviewer.TrainInfoViewer(buffer.data)
            tiv.show()

        builder = flatbuffers.Builder(0)
        builder.Finish(self.modelT.Pack(builder), self.CIRCLE_FILE_IDENTIFIER)
        with open(fname, 'wb') as f:
            f.write(builder.Output())
