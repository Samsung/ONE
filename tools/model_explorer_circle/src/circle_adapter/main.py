"""Adapter module for Circle model"""

from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder


class CicleAdapter(Adapter):
    """Adapter class for Circle models."""
    metadata = AdapterMetadata(id='circle-adapter',
                               name='Circle adapter',
                               description='Circle adapter!',
                               fileExts=['circle'])

    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        graph = graph_builder.Graph(id='main')
        return {'graphs': [graph]}
