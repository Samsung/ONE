"""Adapter module for Circle model"""

from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
from circle_adapter import circle_schema_generated as circle_schema


class CicleAdapter(Adapter):
    """Adapter class for Circle models."""
    metadata = AdapterMetadata(id='circle-adapter',
                               name='Circle adapter',
                               description='Circle adapter!',
                               fileExts=['circle'])

    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self, model_path: str) -> None:
        """Load the model from the given path."""
        with open(model_path, 'rb') as fp:
            model_ = circle_schema.Model.GetRootAsModel(fp.read(), 0)

        self.model = circle_schema.ModelT.InitFromObj(model_)

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert the model to a set of graphs."""
        graph = graph_builder.Graph(id='main')

        self.load_model(model_path)

        return {'graphs': [graph]}
