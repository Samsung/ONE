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
        self.dict_opcode_to_name = {
            v: k for k, v in circle_schema.BuiltinOperator.__dict__.items()}
        # tensor_id -> node_id/output_id
        self.tensor_id_to_src = {}
        self.dict_tensor_type_to_string = {
            v: k for k, v in circle_schema.TensorType.__dict__.items()}

    def load_model(self, model_path: str) -> None:
        """Load the model from the given path."""
        with open(model_path, 'rb') as fp:
            model_ = circle_schema.Model.GetRootAsModel(fp.read(), 0)

        self.model = circle_schema.ModelT.InitFromObj(model_)

    def opcode_to_name(self, opcode: int) -> str:
        """Convert the opcode to its name."""
        return self.dict_opcode_to_name[opcode]

    def tensor_type_to_string(self, tensor_type: int) -> str:
        """Convert a builtin opcode integer to its string representation."""
        return self.dict_tensor_type_to_string[tensor_type].lower()

    def add_map_tensor_to_src(self, tensor_id: int, source_id: int, output_id: int) -> None:
        """Add mapping between tensor id and its source."""
        self.tensor_id_to_src[tensor_id] = f'{source_id}/{output_id}'

    def add_incoming_edge(self,
                          me_node: graph_builder.GraphNode, tensor_id: int, input_id: int) -> None:
        """Add incoming edge to the given node."""
        if tensor_id in self.tensor_id_to_src:
            sid, soid = self.tensor_id_to_src[tensor_id].split('/')
            me_node.incomingEdges.append(
                graph_builder.IncomingEdge(
                    sourceNodeId=sid, sourceNodeOutputId=soid, targetNodeInputId=f'{input_id}')
            )

    def build_graph(self, me_graph: graph_builder.Graph) -> None:
        """Build the graph using the model."""

        sub_graph = self.model.subgraphs[0]

        # Create Input nodes
        input_id = len(sub_graph.operators)
        me_node = graph_builder.GraphNode(
            id=f'{input_id}', label="GraphInputs", namespace="GraphInputs")
        for idx, tensor_id in enumerate(sub_graph.inputs):
            self.add_map_tensor_to_src(tensor_id=tensor_id,
                                       source_id=input_id, output_id=tensor_id)
            # Add metadata to the input node
            tensor = sub_graph.tensors[tensor_id]
            tensor_type = self.tensor_type_to_string(tensor.type)
            tensor_shape = f'{tensor.shape.tolist()}'
            me_node.outputsMetadata.append(
                graph_builder.MetadataItem(
                    id=f'{idx}',
                    attrs=[
                        graph_builder.KeyValue(
                            key='shape', value=tensor_type + tensor_shape),
                        graph_builder.KeyValue(
                            key='tensor_index', value=f'{tensor_id}'),
                        graph_builder.KeyValue(
                            key='tensor_name', value=tensor.name.decode('utf-8')),
                    ],
                )
            )
        me_graph.nodes.append(me_node)

        # Create tensor_id to source map
        for op_id, op in enumerate(sub_graph.operators):
            for i, tensor_id in enumerate(op.outputs):
                self.add_map_tensor_to_src(
                    tensor_id=tensor_id, source_id=op_id, output_id=i)

        # Create pseudo const node for tensors not connected to any operators
        for tensor_id, tensor in enumerate(sub_graph.tensors):
            if (self.tensor_id_to_src.get(tensor_id)) is None:
                self.add_map_tensor_to_src(
                    tensor_id=tensor_id, source_id=input_id + tensor_id, output_id=0)
                me_node = graph_builder.GraphNode(
                    id=f'{input_id + tensor_id}', label='pseudo_const',
                    namespace=tensor.name.decode('utf-8'))
                # Add metadata to the pseudo const node
                tensor_type = self.tensor_type_to_string(tensor.type)
                tensor_shape = f'{tensor.shape.tolist()}'
                me_node.outputsMetadata.append(
                    graph_builder.MetadataItem(
                        id='0',
                        attrs=[
                            graph_builder.KeyValue(
                                key='shape', value=tensor_type + tensor_shape),
                            graph_builder.KeyValue(
                                key='tensor_index', value=f'{tensor_id}'),
                            graph_builder.KeyValue(
                                key='tensor_name', value=tensor.name.decode('utf-8')),
                        ]
                    )
                )
                me_graph.nodes.append(me_node)

        # Create operator nodes
        for idx, op in enumerate(sub_graph.operators):
            name = self.opcode_to_name(
                self.model.operatorCodes[op.opcodeIndex].builtinCode)

            output_tensor_id = op.outputs[0]
            output_tensor = sub_graph.tensors[output_tensor_id]
            # Construct namespace following output tensor's name
            ns = output_tensor.name.decode("utf-8")
            if '/' in ns:
                ns = '/'.join(ns.strip('/').split('/')[:2])

            me_node = graph_builder.GraphNode(
                id=f'{idx}', label=name, namespace=ns)

            # Add ouput tensor metadata
            output_tensor_type = self.tensor_type_to_string(output_tensor.type)
            output_tensor_shape = f'{output_tensor.shape.tolist()}'
            me_node.outputsMetadata.append(
                graph_builder.MetadataItem(
                    id='0',
                    attrs=[
                        graph_builder.KeyValue(
                            key='shape', value=output_tensor_type + output_tensor_shape),
                        graph_builder.KeyValue(
                            key='tensor_index', value=f'{output_tensor_id}'),
                        graph_builder.KeyValue(
                            key='tensor_name', value=output_tensor.name.decode('utf-8')),
                    ]
                )
            )

            # Connect edges from inputs to this operator node
            for i, tensor_id in enumerate(op.inputs):
                if tensor_id < 0:
                    continue
                self.add_incoming_edge(
                    me_node=me_node, tensor_id=tensor_id, input_id=i)
            me_graph.nodes.append(me_node)

        # Create Output nodes
        me_node = graph_builder.GraphNode(
            id=f'{len(me_graph.nodes)}', label="GraphOutputs", namespace="GraphOutputs")
        # Connect edges from inputs to output node
        for i in sub_graph.outputs:
            self.add_incoming_edge(me_node=me_node, tensor_id=i, input_id=i)
        me_graph.nodes.append(me_node)

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert the model to a set of graphs."""
        graph = graph_builder.Graph(id='main')

        self.load_model(model_path)
        self.build_graph(graph)

        return {'graphs': [graph]}
