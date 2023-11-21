/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OperationDumper.h"

#include <string>

#include "util/logging.h"

namespace onert
{
namespace ir
{

using namespace operation;

namespace
{

// Dump all input and output.
// Use this function when there is no special input or(and) output.
void dumpOpGeneric(const Operation &node, const std::string &adding_input = "")
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs() << ") " << adding_input << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs() << ")" << std::endl;
}

void dumpUnaryInputOp(const Operation &node, const std::string &adding_input = "")
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(0) << ") " << adding_input
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void dumpConvOp(const Operation &node, const std::string &padding_type)
{
  VERBOSE(LIR) << "* " << node.name() << "(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(Conv2D::Input::INPUT) << ") Kernel("
               << node.getInputs().at(Conv2D::Input::KERNEL) << ") Bias("
               << node.getInputs().at(Conv2D::Input::BIAS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}
} // namespace

OperationDumper::OperationDumper(const std::string &start_msg)
{
  VERBOSE(LIR) << start_msg << std::endl;
}

void OperationDumper::visit(const ArgMinMax &node)
{
  std::string min_max = node.param().is_arg_max ? "(Max)" : "(Min)";
  VERBOSE(LIR) << "* " << node.name() << min_max << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ArgMinMax::INPUT) << ") Axis("
               << node.getInputs().at(ArgMinMax::AXIS) << ") " << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const BatchToSpaceND &node)
{
  std::string block_size =
    "BlockSize(" + std::to_string(node.getInputs().at(BatchToSpaceND::Input::BLOCK_SIZE).value()) +
    ")";
  dumpOpGeneric(node, block_size);
}

void OperationDumper::visit(const BCQFullyConnected &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(BCQFullyConnected::Input::INPUT)
               << ") WeightsBinary("
               << node.getInputs().at(BCQFullyConnected::Input::WEIGHTS_BINARY)
               << ") WeightsScales("
               << node.getInputs().at(BCQFullyConnected::Input::WEIGHTS_SCALES)
               << ") WeightsClusters("
               << node.getInputs().at(BCQFullyConnected::Input::WEIGHTS_CLUSTERS) << ") Bias("
               << node.getInputs().at(BCQFullyConnected::Input::BIAS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const BinaryArithmetic &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const operation::BroadcastTo &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Comparison &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Concat &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Conv2D &node)
{
  std::string padding_type =
    node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  dumpConvOp(node, padding_type);
}

void OperationDumper::visit(const ConvertFp16ToFp32 &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const ConvertFp32ToFp16 &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const DepthToSpace &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const DepthwiseConv2D &node)
{
  std::string padding_type =
    node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  dumpConvOp(node, padding_type);
}

void OperationDumper::visit(const ElementwiseActivation &node)
{
  std::string params;
  if (node.param().op_type == ElementwiseActivation::Type::RELU)
  {
    params = " lower value(" + std::to_string(node.param().alpha) + ") upper value(" +
             std::to_string(node.param().beta) + ")";
  }
  else if (node.param().op_type == ElementwiseActivation::Type::LEAKY_RELU)
  {
    params = " alpha value(" + std::to_string(node.param().alpha) + ")";
  }
  dumpOpGeneric(node, params);
}

void OperationDumper::visit(const ElementwiseBinary &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const ElementwiseUnary &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const EmbeddingLookup &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Lookups(" << node.getInputs().at(EmbeddingLookup::Input::LOOKUPS)
               << ") VALUES(" << node.getInputs().at(EmbeddingLookup::Input::VALUES) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ExpandDims &node)
{
  std::string axis =
    "AXIS(" + std::to_string(node.getInputs().at(ExpandDims::Input::AXIS).value()) + ")";
  dumpUnaryInputOp(node, axis);
}

void OperationDumper::visit(const Fill &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Shape(" << node.getInputs().at(Fill::Input::SHAPE) << ") Value("
               << node.getInputs().at(Fill::Input::VALUE) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const FullyConnected &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ArgMinMax::INPUT) << ") Weight("
               << node.getInputs().at(FullyConnected::Input::WEIGHT) << ") Bias("
               << node.getInputs().at(FullyConnected::Input::BIAS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Gather &node)
{
  std::string indices =
    "Indices(" + std::to_string(node.getInputs().at(Gather::Input::INDICES).value()) + ")";
  dumpUnaryInputOp(node, indices);
}

void OperationDumper::visit(const HashtableLookup &node)
{
  VERBOSE(LIR) << "* HashTableLookup" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Lookups(" << node.getInputs().at(HashtableLookup::Input::LOOKUPS)
               << ") Keys(" << node.getInputs().at(HashtableLookup::Input::KEYS) << ") Values("
               << node.getInputs().at(HashtableLookup::Input::VALUES) << ")" << std::endl;
  VERBOSE(LIR) << "  - Outputs : Output(" << node.getInputs().at(HashtableLookup::Output::OUTPUT)
               << ") Hits(" << node.getInputs().at(HashtableLookup::Output::HITS) << ")"
               << std::endl;
}

void OperationDumper::visit(const InstanceNorm &node)
{
  std::string inputs =
    "Gamma(" + std::to_string(node.getInputs().at(InstanceNorm::Input::GAMMA).value()) + ") Beta(" +
    std::to_string(node.getInputs().at(InstanceNorm::Input::BETA).value()) + ")";
  dumpUnaryInputOp(node, inputs);
}

void OperationDumper::visit(const L2Normalization &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const LocalResponseNormalization &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Loss &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << " - Inputs : Prediction(" << node.getInputs().at(Loss::Input::Y_PRED) << ") True("
               << node.getInputs().at(Loss::Input::Y_TRUE) << ")" << std::endl;
  VERBOSE(LIR) << " - Outputs : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const LSTM &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR)
    << "  - Inputs : Input(" << node.getInputs().at(LSTM::Input::INPUT)
    << ") Input To Input Weights(" << node.getInputs().at(LSTM::Input::INPUT_TO_INPUT_WEIGHTS)
    << ") Input To Forget Weights(" << node.getInputs().at(LSTM::Input::INPUT_TO_FORGET_WEIGHTS)
    << ") Input To Cell Weights(" << node.getInputs().at(LSTM::Input::INPUT_TO_CELL_WEIGHTS)
    << ") Input To Output Weights(" << node.getInputs().at(LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS)
    << ") Recurrent To Input Weights("
    << node.getInputs().at(LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS)
    << ") Recurrent To Forget Weights("
    << node.getInputs().at(LSTM::Input::RECURRENT_TO_FORGET_WEIGHTS)
    << ") Recurrent To Cell Weights(" << node.getInputs().at(LSTM::Input::RECURRENT_TO_CELL_WEIGHTS)
    << ") Recurrent To Output Weights("
    << node.getInputs().at(LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS) << ") Cell To Input Weights("
    << node.getInputs().at(LSTM::Input::CELL_TO_INPUT_WEIGHTS) << ") Cell To Forget Weights("
    << node.getInputs().at(LSTM::Input::CELL_TO_FORGET_WEIGHTS) << ") Cell To OUTPUT Weights("
    << node.getInputs().at(LSTM::Input::CELL_TO_OUTPUT_WEIGHTS) << ") Input Gate Bias("
    << node.getInputs().at(LSTM::Input::INPUT_GATE_BIAS) << ") Forget Gate Bias("
    << node.getInputs().at(LSTM::Input::FORGET_GATE_BIAS) << ") Cell Bias("
    << node.getInputs().at(LSTM::Input::CELL_BIAS) << ") Output Gate Bias("
    << node.getInputs().at(LSTM::Input::OUTPUT_GATE_BIAS) << ") Projection Weights("
    << node.getInputs().at(LSTM::Input::PROJECTION_WEIGHTS) << ") Projection Bias("
    << node.getInputs().at(LSTM::Input::PROJECTION_BIAS) << ") Output State In("
    << node.getInputs().at(LSTM::Input::OUTPUT_STATE_IN) << ") Cell State In("
    << node.getInputs().at(LSTM::Input::CELL_STATE_IN);
  if (node.getInputs().size() == 24)
  {
    VERBOSE(LIR) << ") Input Layer Normalization Weights("
                 << node.getInputs().at(LSTM::Input::INPUT_LAYER_NORMALIZATION_WEIGHTS)
                 << ") Forget Layer Normalization Weights("
                 << node.getInputs().at(LSTM::Input::FORGET_LAYER_NORMALIZATION_WEIGHTS)
                 << ") Cell Layer Normalization Weights("
                 << node.getInputs().at(LSTM::Input::CELL_LAYER_NORMALIZATION_WEIGHTS)
                 << ") Ouput Layer Normalization Weights("
                 << node.getInputs().at(LSTM::Input::OUTPUT_LAYER_NORMALIZATION_WEIGHTS);
  }
  VERBOSE(LIR) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Scratch Buffer("
               << node.getOutputs().at(LSTM::Output::SCRATCH_BUFFER) << ") Output State Out("
               << node.getOutputs().at(LSTM::Output::OUTPUT_STATE_OUT) << ") Cell State Out("
               << node.getOutputs().at(LSTM::Output::CELL_STATE_OUT) << ") Output("
               << node.getOutputs().at(LSTM::Output::OUTPUT) << ")" << std::endl;
}

void OperationDumper::visit(const Pack &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Pad &node)
{
  std::string pad = "Pad(" + std::to_string(node.getInputs().at(Pad::Input::PAD).value()) + ")";
  dumpUnaryInputOp(node, pad);
}

void OperationDumper::visit(const Permute &node)
{
  std::string permute_type = "Unknown";
  switch (node.getPermuteType())
  {
    case Permute::Type::COPY:
      permute_type = "Copy";
      break;
    case Permute::Type::NHWC_TO_NCHW:
      permute_type = "NHWC to NCHW";
      break;
    case Permute::Type::NCHW_TO_NHWC:
      permute_type = "NCHW to NHWC";
      break;
  }

  VERBOSE(LIR) << "* Permute(" + permute_type + ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(0) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Pool2D &node)
{
  std::string padding_type =
    node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* " << node.name() << "(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(Pool2D::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Pow &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const PReLU &node)
{
  std::string alpha =
    "Alpha(" + std::to_string(node.getInputs().at(PReLU::Input::ALPHA).value()) + ")";
  dumpOpGeneric(node, alpha);
}

void OperationDumper::visit(const Rank &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Reduce &node) { dumpUnaryInputOp(node); }

void OperationDumper::visit(const Reshape &node)
{
  // optional param
  std::string shape =
    node.getInputs().size() == 2
      ? "Shape(" + std::to_string(node.getInputs().at(Reshape::Input::SHAPE).value()) + ")"
      : "Shape(not provided)";
  dumpUnaryInputOp(node, shape);
}

void OperationDumper::visit(const ResizeBilinear &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const ResizeNearestNeighbor &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Reverse &node)
{
  std::string axis =
    "Axis(" + std::to_string(node.getInputs().at(Reverse::Input::AXIS).value()) + ")";
  dumpUnaryInputOp(node, axis);
}

void OperationDumper::visit(const RNN &node)
{
  VERBOSE(LIR) << "* RNN" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(RNN::Input::INPUT) << ") Weights("
               << node.getInputs().at(RNN::Input::WEIGHTS) << ") Recurrent Weights("
               << node.getInputs().at(RNN::Input::RECURRENT_WEIGHTS) << ") Bias("
               << node.getInputs().at(RNN::Input::BIAS) << ") Hidden State("
               << node.getInputs().at(RNN::Input::HIDDEN_STATE_IN) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(RNN::Output::OUTPUT)
               << ") Hidden State(" << node.getInputs().at(RNN::Output::HIDDEN_STATE_OUT) << ")"
               << std::endl;
}

void OperationDumper::visit(const Range &node)
{
  VERBOSE(LIR) << "* Range" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Start(" << node.getInputs().at(Range::Input::START) << ")"
               << " Limit(" << node.getInputs().at(Range::Input::LIMIT) << ")"
               << " Delta(" << node.getInputs().at(Range::Input::DELTA) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Select &node)
{
  VERBOSE(LIR) << "* Select" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Condition(" << node.getInputs().at(Select::Input::CONDITION) << ")"
               << " Input_X(" << node.getInputs().at(Select::Input::INPUT_TRUE) << ")"
               << " Input_Y(" << node.getInputs().at(Select::Input::INPUT_FALSE) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ir::operation::Shape &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Softmax &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const SpaceToBatchND &node)
{
  std::string inputs =
    "BlockSize(" + std::to_string(node.getInputs().at(SpaceToBatchND::Input::BLOCK_SIZE).value()) +
    ") Paddings(" + std::to_string(node.getInputs().at(SpaceToBatchND::Input::PADDINGS).value()) +
    ")";
  dumpUnaryInputOp(node, inputs);
}

void OperationDumper::visit(const SpaceToDepth &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Split &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const SquaredDifference &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const StatelessRandomUniform &node)
{
  VERBOSE(LIR) << "* StatelessRandomUniform" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Shape(" << node.getInputs().at(StatelessRandomUniform::Input::SHAPE)
               << " Seed(" << node.getInputs().at(StatelessRandomUniform::Input::SEED) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Squeeze &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Slice &node) { dumpUnaryInputOp(node); }

void OperationDumper::visit(const StridedSlice &node) { dumpUnaryInputOp(node); }

void OperationDumper::visit(const Tile &node)
{
  std::string multiples =
    "Multiples(" + std::to_string(node.getInputs().at(Tile::Input::MULTIPLES).value()) + ")";
  dumpUnaryInputOp(node, multiples);
}

void OperationDumper::visit(const TopKV2 &node)
{
  VERBOSE(LIR) << "* TopKV2" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(TopKV2::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Outputs : Values(" << node.getOutputs().at(TopKV2::Output::OUTPUT_VALUES)
               << ") Indices(" << node.getOutputs().at(TopKV2::Output::OUTPUT_INDICES) << ")"
               << std::endl;
}

void OperationDumper::visit(const TransposeConv &node)
{
  std::string padding_type =
    node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* TransposeConv(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Output Shape("
               << node.getInputs().at(TransposeConv::Input::OUTPUT_SHAPE) << ") KERNEL("
               << node.getInputs().at(TransposeConv::Input::KERNEL) << ") IFM("
               << node.getInputs().at(TransposeConv::Input::INPUT) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Transpose &node) { dumpOpGeneric(node); }

void OperationDumper::visit(const Unpack &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Unpack::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Outputs(" << node.getOutputs() << ")" << std::endl;
}

void OperationDumper::visit(const OneHot &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : "
               << "Indices(" << node.getInputs().at(OneHot::Input::INDICES) << ") " << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const If &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : "
               << "Then subgraph (" << node.param().then_subg_index << ") Else subgraph ("
               << node.param().else_subg_index << ") Inputs(" << node.getInputs() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Outputs(" << node.getOutputs() << ")" << std::endl;
}

void OperationDumper::visit(const While &node)
{
  VERBOSE(LIR) << "* " << node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : "
               << "Cond subgraph (" << node.param().cond_subg_index << ") Body subgraph ("
               << node.param().body_subg_index << ") Inputs(" << node.getInputs() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Outputs(" << node.getOutputs() << ")" << std::endl;
}

} // namespace ir
} // namespace onert
