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

OperationDumper::OperationDumper(const std::string &start_msg)
{
  VERBOSE(LIR) << start_msg << std::endl;
}

void OperationDumper::visit(const ArgMax &node)
{
  VERBOSE(LIR) << "* ArgMax" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ArgMax::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const BatchToSpaceND &node)
{
  VERBOSE(LIR) << "* BatchToSpaceND" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(BatchToSpaceND::Input::INPUT) << ")"
               << " BlockSize(" << node.getInputs().at(BatchToSpaceND::Input::BLOCK_SIZE) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const BinaryArithmetic &node)
{
  VERBOSE(LIR) << "* " + node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(BinaryArithmetic::Input::LHS) << ", "
               << node.getInputs().at(BinaryArithmetic::Input::RHS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const operation::BroadcastTo &node)
{
  VERBOSE(LIR) << "* BroadcastTo" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(BroadcastTo::Input::INPUT) << ", "
               << node.getInputs().at(BroadcastTo::Input::SHAPE) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Comparison &node)
{
  VERBOSE(LIR) << "* Comparison" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Comparison::Input::INPUT0) << ", "
               << node.getInputs().at(Comparison::Input::INPUT1) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Concat &node)
{
  VERBOSE(LIR) << "* Concat" << std::endl;
  std::string inputs;
  for (auto i : node.getInputs())
  {
    inputs += std::to_string(i.value()) + ",";
  }
  VERBOSE(LIR) << "  - Inputs : IFM(" << inputs << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Conv2D &node)
{
  std::string padding_type =
      node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* Conv2D(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(Conv2D::Input::INPUT) << ") Kernel("
               << node.getInputs().at(Conv2D::Input::KERNEL) << ") Bias("
               << node.getInputs().at(Conv2D::Input::BIAS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ConvertFp16ToFp32 &node)
{
  VERBOSE(LIR) << "* ConvertFp16ToFp32" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ConvertFp16ToFp32::Input::INPUT)
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ConvertFp32ToFp16 &node)
{
  VERBOSE(LIR) << "* ConvertFp32ToFp16" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ConvertFp32ToFp16::Input::INPUT)
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const DepthToSpace &node)
{
  VERBOSE(LIR) << "* DepthToSpace" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(DepthToSpace::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const DepthwiseConv2D &node)
{
  std::string padding_type =
      node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* DepthwiseConv2D(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(DepthwiseConv2D::Input::INPUT)
               << ") Kernel(" << node.getInputs().at(DepthwiseConv2D::Input::KERNEL) << ") Bias("
               << node.getInputs().at(DepthwiseConv2D::Input::BIAS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ElementwiseActivation &node)
{
  VERBOSE(LIR) << "* " + node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ElementwiseActivation::Input::INPUT)
               << ")";
  if (node.param().op_type == ElementwiseActivation::Type::RELU)
  {
    VERBOSE(LIR) << " lower value(" << node.param().alpha << ") upper value(" << node.param().beta
                 << ")";
  }
  VERBOSE(LIR) << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ElementwiseBinary &node)
{
  VERBOSE(LIR) << "* " + node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ElementwiseBinary::Input::LHS)
               << ", " << node.getInputs().at(ElementwiseBinary::Input::RHS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ElementwiseUnary &node)
{
  VERBOSE(LIR) << "* " + node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ElementwiseUnary::Input::INPUT)
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const EmbeddingLookup &node)
{
  VERBOSE(LIR) << "* EmbeddingLookup" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Lookups(" << node.getInputs().at(EmbeddingLookup::Input::LOOKUPS)
               << ") VALUES(" << node.getInputs().at(EmbeddingLookup::Input::VALUES) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ExpandDims &node)
{
  VERBOSE(LIR) << "* ExpandDims" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ExpandDims::Input::INPUT)
               << ") AXIS(" << node.getInputs().at(ExpandDims::Input::AXIS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const FullyConnected &node)
{
  VERBOSE(LIR) << "* FullyConnected" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(FullyConnected::Input::INPUT)
               << ") Weight(" << node.getInputs().at(FullyConnected::Input::WEIGHT) << ") Bias("
               << node.getInputs().at(FullyConnected::Input::BIAS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Gather &node)
{
  VERBOSE(LIR) << "* Gather" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Gather::Input::INPUT) << ") Indices("
               << node.getInputs().at(Gather::Input::INDICES) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
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
  VERBOSE(LIR) << "* InstanceNorm" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(InstanceNorm::Input::INPUT)
               << ") Gamma(" << node.getInputs().at(InstanceNorm::Input::GAMMA) << ") Beta("
               << node.getInputs().at(InstanceNorm::Input::BETA) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const L2Normalization &node)
{
  VERBOSE(LIR) << "* L2Normalization" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(L2Normalization::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const LocalResponseNormalization &node)
{
  VERBOSE(LIR) << "* LocalResponseNormalization" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input("
               << node.getInputs().at(LocalResponseNormalization::Input::INPUT) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const LSTM &node)
{
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
      << ") Recurrent To Cell Weights("
      << node.getInputs().at(LSTM::Input::RECURRENT_TO_CELL_WEIGHTS)
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
      << node.getInputs().at(LSTM::Input::CELL_STATE_IN) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Scratch Buffer("
               << node.getOutputs().at(LSTM::Output::SCRATCH_BUFFER) << ") Output State Out("
               << node.getInputs().at(LSTM::Output::OUTPUT_STATE_OUT) << ") Cell State Out("
               << node.getInputs().at(LSTM::Output::CELL_STATE_OUT) << ") Output("
               << node.getInputs().at(LSTM::Output::OUTPUT) << ")" << std::endl;
}

void OperationDumper::visit(const Pack &node)
{
  VERBOSE(LIR) << "* Pack" << std::endl;
  std::string inputs;
  const auto &input_indices = node.getInputs();
  for (auto it = std::begin(input_indices); it != std::end(input_indices); ++it)
  {
    inputs += std::to_string(it->value());
    if (std::next(it) != std::end(input_indices))
      inputs += ", ";
  }
  VERBOSE(LIR) << "  - Inputs : Inputs(" << inputs << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Pad &node)
{
  VERBOSE(LIR) << "* Pad" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Pad::Input::INPUT) << ") Pad("
               << node.getInputs().at(Pad::Input::PAD) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
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

void OperationDumper::visit(const Pow &node)
{
  VERBOSE(LIR) << "* Pow" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Pow::Input::LHS) << ", "
               << node.getInputs().at(Pow::Input::RHS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const PReLU &node)
{
  VERBOSE(LIR) << "* PReLU" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(PReLU::Input::INPUT) << ") Alpha("
               << node.getInputs().at(PReLU::Input::ALPHA) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Reduce &node)
{
  VERBOSE(LIR) << "* " + node.name() << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Reduce::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Reshape &node)
{
  VERBOSE(LIR) << "* Reshape" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Reshape::Input::INPUT) << ")";
  // optional param
  if (node.getInputs().size() == 2)
  {
    VERBOSE(LIR) << " Shape(" << node.getInputs().at(Reshape::Input::SHAPE) << ")";
  }
  else
  {
    VERBOSE(LIR) << " Shape(not provided)";
  }
  VERBOSE(LIR) << std::endl;

  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ResizeBilinear &node)
{
  VERBOSE(LIR) << "* ResizeBilinear" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ResizeBilinear::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Reverse &node)
{
  VERBOSE(LIR) << "* Reverse" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Reverse::Input::INPUT) << ") Axis("
               << node.getInputs().at(Reverse::Input::AXIS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
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
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Range::Input::START) << ")"
               << " Limit(" << node.getInputs().at(Range::Input::LIMIT) << ")"
               << " Delta(" << node.getInputs().at(Range::Input::DELTA) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Select &node)
{
  VERBOSE(LIR) << "* Select" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Select::Input::CONDITION) << ")"
               << " Input_X(" << node.getInputs().at(Select::Input::INPUT_TRUE) << ")"
               << " Input_Y(" << node.getInputs().at(Select::Input::INPUT_FALSE) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const ir::operation::Shape &node)
{
  VERBOSE(LIR) << "* Shape" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ir::operation::Shape::Input::INPUT)
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Softmax &node)
{
  VERBOSE(LIR) << "* Softmax" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Softmax::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const SpaceToBatchND &node)
{
  VERBOSE(LIR) << "* SpaceToBatchND" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(SpaceToBatchND::Input::INPUT)
               << ") BlockSize(" << node.getInputs().at(SpaceToBatchND::Input::BLOCK_SIZE)
               << ") Paddings(" << node.getInputs().at(SpaceToBatchND::Input::PADDINGS) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const SpaceToDepth &node)
{
  VERBOSE(LIR) << "* SpaceToDepth" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(SpaceToDepth::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Split &node)
{
  VERBOSE(LIR) << "* Split" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Split::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const SquaredDifference &node)
{
  VERBOSE(LIR) << "* SquaredDifference" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(SquaredDifference::Input::LHS)
               << ", " << node.getInputs().at(SquaredDifference::Input::RHS) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const StatelessRandomUniform &node)
{
  VERBOSE(LIR) << "* StatelessRandomUniform" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(StatelessRandomUniform::Input::SHAPE)
               << ", " << node.getInputs().at(StatelessRandomUniform::Input::SEED) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Squeeze &node)
{
  VERBOSE(LIR) << "* Squeeze" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Squeeze::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Slice &node)
{
  VERBOSE(LIR) << "* Slice" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Slice::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const StridedSlice &node)
{
  VERBOSE(LIR) << "* StridedSlice" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(StridedSlice::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Tile &node)
{
  VERBOSE(LIR) << "* Tile" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Tile::Input::INPUT) << ", "
               << node.getInputs().at(Tile::Input::MULTIPLES) << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
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

void OperationDumper::visit(const Transpose &node)
{
  VERBOSE(LIR) << "* Transpose" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Transpose::Input::INPUT) << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const Unpack &node)
{
  VERBOSE(LIR) << "* Unpack" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Unpack::Input::INPUT) << ")"
               << std::endl;
  std::string outputs;
  const auto &output_indices = node.getOutputs();
  for (auto it = std::begin(output_indices); it != std::end(output_indices); ++it)
  {
    outputs += std::to_string(it->value());
    if (std::next(it) != std::end(output_indices))
      outputs += ", ";
  }
  VERBOSE(LIR) << "  - Outputs : Outputs(" << outputs << ")" << std::endl;
}

void OperationDumper::visit(const OneHot &node)
{
  VERBOSE(LIR) << "* OneHot" << std::endl;
  VERBOSE(LIR) << "  - Inputs : "
               << "Indices(" << node.getInputs().at(OneHot::Input::INDICES) << ") " << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0) << ")" << std::endl;
}

void OperationDumper::visit(const If &node)
{
  VERBOSE(LIR) << "* If" << std::endl;
  std::string inputs;
  const auto &input_indices = node.getInputs();
  for (auto it = std::begin(input_indices); it != std::end(input_indices); ++it)
  {
    inputs += std::to_string(it->value());
    if (std::next(it) != std::end(input_indices))
      inputs += ", ";
  }
  VERBOSE(LIR) << "  - Inputs : "
               << "Then subgraph (" << node.param().then_subg_index << ") Else subgraph ("
               << node.param().else_subg_index << ") Inputs(" << inputs << ")" << std::endl;
  std::string outputs;
  const auto &output_indices = node.getOutputs();
  for (auto it = std::begin(output_indices); it != std::end(output_indices); ++it)
  {
    outputs += std::to_string(it->value());
    if (std::next(it) != std::end(output_indices))
      outputs += ", ";
  }
  VERBOSE(LIR) << "  - Output : Outputs(" << outputs << ")" << std::endl;
}

void OperationDumper::visit(const While &node)
{
  VERBOSE(LIR) << "* While" << std::endl;
  std::string inputs;
  const auto &input_indices = node.getInputs();
  for (auto it = std::begin(input_indices); it != std::end(input_indices); ++it)
  {
    inputs += std::to_string(it->value());
    if (std::next(it) != std::end(input_indices))
      inputs += ", ";
  }
  VERBOSE(LIR) << "  - Inputs : "
               << "Cond subgraph (" << node.param().cond_subg_index << ") Body subgraph ("
               << node.param().cond_subg_index << ") Inputs(" << inputs << ")" << std::endl;
  std::string outputs;
  const auto &output_indices = node.getOutputs();
  for (auto it = std::begin(output_indices); it != std::end(output_indices); ++it)
  {
    outputs += std::to_string(it->value());
    if (std::next(it) != std::end(output_indices))
      outputs += ", ";
  }
  VERBOSE(LIR) << "  - Output : Outputs(" << outputs << ")" << std::endl;
}

} // namespace ir
} // namespace onert
