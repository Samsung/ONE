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

namespace neurun
{
namespace ir
{

using namespace operation;

void OperationDumper::visit(const Abs &node)
{
  VERBOSE(LIR) << "* Abs" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Abs::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Add &node)
{
  VERBOSE(LIR) << "* Add" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Add::Input::LHS).value() << ", "
               << node.getInputs().at(Add::Input::RHS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ArgMax &node)
{
  VERBOSE(LIR) << "* ArgMax" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ArgMax::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const AvgPool2D &node)
{
  VERBOSE(LIR) << "* AvgPool2D(Implicit)" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(AvgPool2D::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Cast &node)
{
  VERBOSE(LIR) << "* Cast" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Cast::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Comparison &node)
{
  VERBOSE(LIR) << "* Comparison" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Comparison::Input::INPUT0).value()
               << ", " << node.getInputs().at(Comparison::Input::INPUT1).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
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
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Conv2D &node)
{
  std::string padding_type =
      node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* Conv2D(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(Conv2D::Input::INPUT).value()
               << ") Kernel(" << node.getInputs().at(Conv2D::Input::KERNEL).value() << ") Bias("
               << node.getInputs().at(Conv2D::Input::BIAS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const DepthToSpace &node)
{
  VERBOSE(LIR) << "* DepthToSpace" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(DepthToSpace::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const DepthwiseConv2D &node)
{
  std::string padding_type =
      node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* DepthwiseConv2D(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(DepthwiseConv2D::Input::INPUT).value()
               << ") Kernel(" << node.getInputs().at(DepthwiseConv2D::Input::KERNEL).value()
               << ") Bias(" << node.getInputs().at(DepthwiseConv2D::Input::BIAS).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Dequantize &node)
{
  VERBOSE(LIR) << "* Dequantize" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Dequantize::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Div &node)
{
  VERBOSE(LIR) << "* Div" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Div::Input::LHS).value() << ", "
               << node.getInputs().at(Div::Input::RHS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const EmbeddingLookup &node)
{
  VERBOSE(LIR) << "* EmbeddingLookup" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Lookups("
               << node.getInputs().at(EmbeddingLookup::Input::LOOKUPS).value() << ") VALUES("
               << node.getInputs().at(EmbeddingLookup::Input::VALUES).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Exp &node)
{
  VERBOSE(LIR) << "* Exp" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Exp::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Floor &node)
{
  VERBOSE(LIR) << "* Floor" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Floor::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const FullyConnected &node)
{
  VERBOSE(LIR) << "* FullyConnected" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(FullyConnected::Input::INPUT).value()
               << ") Weight(" << node.getInputs().at(FullyConnected::Input::WEIGHT).value()
               << ") Bias(" << node.getInputs().at(FullyConnected::Input::BIAS).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Gather &node)
{
  VERBOSE(LIR) << "* Gather" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Gather::Input::INPUT).value()
               << ") Indices(" << node.getInputs().at(Gather::Input::INDICES).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const HashtableLookup &node)
{
  VERBOSE(LIR) << "* HashTableLookup" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Lookups("
               << node.getInputs().at(HashtableLookup::Input::LOOKUPS).value() << ") Keys("
               << node.getInputs().at(HashtableLookup::Input::KEYS).value() << ") Values("
               << node.getInputs().at(HashtableLookup::Input::VALUES).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Outputs : Output("
               << node.getInputs().at(HashtableLookup::Output::OUTPUT).value() << ") Hits("
               << node.getInputs().at(HashtableLookup::Output::HITS).value() << ")" << std::endl;
}

void OperationDumper::visit(const InstanceNorm &node)
{
  VERBOSE(LIR) << "* InstanceNorm" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(InstanceNorm::Input::INPUT).value()
               << ") Gamma(" << node.getInputs().at(InstanceNorm::Input::GAMMA).value() << ") Beta("
               << node.getInputs().at(InstanceNorm::Input::BETA).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const L2Normalization &node)
{
  VERBOSE(LIR) << "* L2Normalization" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input("
               << node.getInputs().at(L2Normalization::Input::INPUT).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const L2Pool2D &node)
{
  VERBOSE(LIR) << "* L2Pool2D" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(L2Pool2D::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const LocalResponseNormalization &node)
{
  VERBOSE(LIR) << "* LocalResponseNormalization" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input("
               << node.getInputs().at(LocalResponseNormalization::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const LSTM &node)
{
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(LSTM::Input::INPUT).value()
               << ") Input To Input Weights("
               << node.getInputs().at(LSTM::Input::INPUT_TO_INPUT_WEIGHTS).value()
               << ") Input To Forget Weights("
               << node.getInputs().at(LSTM::Input::INPUT_TO_FORGET_WEIGHTS).value()
               << ") Input To Cell Weights("
               << node.getInputs().at(LSTM::Input::INPUT_TO_CELL_WEIGHTS).value()
               << ") Input To Output Weights("
               << node.getInputs().at(LSTM::Input::INPUT_TO_OUTPUT_WEIGHTS).value()
               << ") Recurrent To Input Weights("
               << node.getInputs().at(LSTM::Input::RECURRENT_TO_INPUT_WEIGHTS).value()
               << ") Recurrent To Forget Weights("
               << node.getInputs().at(LSTM::Input::RECURRENT_TO_FORGET_WEIGHTS).value()
               << ") Recurrent To Cell Weights("
               << node.getInputs().at(LSTM::Input::RECURRENT_TO_CELL_WEIGHTS).value()
               << ") Recurrent To Output Weights("
               << node.getInputs().at(LSTM::Input::RECURRENT_TO_OUTPUT_WEIGHTS).value()
               << ") Cell To Input Weights("
               << node.getInputs().at(LSTM::Input::CELL_TO_INPUT_WEIGHTS).value()
               << ") Cell To Forget Weights("
               << node.getInputs().at(LSTM::Input::CELL_TO_FORGET_WEIGHTS).value()
               << ") Cell To OUTPUT Weights("
               << node.getInputs().at(LSTM::Input::CELL_TO_OUTPUT_WEIGHTS).value()
               << ") Input Gate Bias(" << node.getInputs().at(LSTM::Input::INPUT_GATE_BIAS).value()
               << ") Forget Gate Bias("
               << node.getInputs().at(LSTM::Input::FORGET_GATE_BIAS).value() << ") Cell Bias("
               << node.getInputs().at(LSTM::Input::CELL_BIAS).value() << ") Output Gate Bias("
               << node.getInputs().at(LSTM::Input::OUTPUT_GATE_BIAS).value()
               << ") Projection Weights("
               << node.getInputs().at(LSTM::Input::PROJECTION_WEIGHTS).value()
               << ") Projection Bias(" << node.getInputs().at(LSTM::Input::PROJECTION_BIAS).value()
               << ") Output State In(" << node.getInputs().at(LSTM::Input::OUTPUT_STATE_IN).value()
               << ") Cell State In(" << node.getInputs().at(LSTM::Input::CELL_STATE_IN).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Scratch Buffer("
               << node.getOutputs().at(LSTM::Output::SCRATCH_BUFFER).value()
               << ") Output State Out("
               << node.getInputs().at(LSTM::Output::OUTPUT_STATE_OUT).value() << ") Cell State Out("
               << node.getInputs().at(LSTM::Output::CELL_STATE_OUT).value() << ") Output("
               << node.getInputs().at(LSTM::Output::OUTPUT).value() << ")" << std::endl;
}

void OperationDumper::visit(const LogicalAnd &node)
{
  VERBOSE(LIR) << "* LogicalAnd" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(LogicalAnd::Input::INPUT0).value()
               << ", " << node.getInputs().at(LogicalAnd::Input::INPUT1).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const LogicalNot &node)
{
  VERBOSE(LIR) << "* LogicalNot" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(LogicalNot::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const LogicalOr &node)
{
  VERBOSE(LIR) << "* LogicalOr" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(LogicalOr::Input::INPUT0).value()
               << ", " << node.getInputs().at(LogicalOr::Input::INPUT1).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Logistic &node)
{
  VERBOSE(LIR) << "* Logistic" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Logistic::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const MaxPool2D &node)
{
  std::string padding_type =
      node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* MaxPool2D(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : IFM(" << node.getInputs().at(MaxPool2D::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Mean &node)
{
  VERBOSE(LIR) << "* Mean" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Mean::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Mul &node)
{
  VERBOSE(LIR) << "* Mul" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Mul::Input::LHS).value() << ", "
               << node.getInputs().at(Mul::Input::RHS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Neg &node)
{
  VERBOSE(LIR) << "* Neg" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Neg::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
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
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
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
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(0).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const PReLU &node)
{
  VERBOSE(LIR) << "* PReLU" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(PReLU::Input::INPUT).value()
               << ") Alpha(" << node.getInputs().at(PReLU::Input::ALPHA).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ReduceMax &node)
{
  VERBOSE(LIR) << "* ReduceMax" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ReduceMax::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ReduceMin &node)
{
  VERBOSE(LIR) << "* ReduceMin" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ReduceMin::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ReduceSum &node)
{
  VERBOSE(LIR) << "* ReduceSum" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ReduceSum::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ReLU &node)
{
  VERBOSE(LIR) << "* ReLU" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ReLU::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ReLU1 &node)
{
  VERBOSE(LIR) << "* ReLU1" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ReLU1::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ReLU6 &node)
{
  VERBOSE(LIR) << "* ReLU6" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ReLU6::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Reshape &node)
{
  VERBOSE(LIR) << "* Reshape" << std::endl;
  // TODO The shape index should be "node.getInputs().at(1).value()" but not valid for now
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Reshape::Input::INPUT).value()
               << ") Shape("
               << "?"
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const ResizeBilinear &node)
{
  VERBOSE(LIR) << "* ResizeBilinear" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(ResizeBilinear::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const RNN &node)
{
  VERBOSE(LIR) << "* RNN" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(RNN::Input::INPUT).value()
               << ") Weights" << node.getInputs().at(RNN::Input::WEIGHTS).value()
               << ") Recurrent Weights"
               << node.getInputs().at(RNN::Input::RECURRENT_WEIGHTS).value() << ") Bias"
               << node.getInputs().at(RNN::Input::BIAS).value() << ") Hidden State"
               << node.getInputs().at(RNN::Input::HIDDEN_STATE_IN).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(RNN::Output::OUTPUT).value()
               << ") Hidden State" << node.getInputs().at(RNN::Output::HIDDEN_STATE_OUT).value()
               << ")" << std::endl;
}

void OperationDumper::visit(const RSQRT &node)
{
  VERBOSE(LIR) << "* RSQRT" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(RSQRT::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Softmax &node)
{
  VERBOSE(LIR) << "* Softmax" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Softmax::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const SpaceToDepth &node)
{
  VERBOSE(LIR) << "* SpaceToDepth" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(SpaceToDepth::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Split &node)
{
  VERBOSE(LIR) << "* Split" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Split::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const SQRT &node)
{
  VERBOSE(LIR) << "* SQRT" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(SQRT::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const SquaredDifference &node)
{
  VERBOSE(LIR) << "* SquaredDifference" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input("
               << node.getInputs().at(SquaredDifference::Input::LHS).value() << ", "
               << node.getInputs().at(SquaredDifference::Input::RHS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Squeeze &node)
{
  VERBOSE(LIR) << "* Squeeze" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Squeeze::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Slice &node)
{
  VERBOSE(LIR) << "* Slice" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Slice::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const StridedSlice &node)
{
  VERBOSE(LIR) << "* StridedSlice" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(StridedSlice::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Sub &node)
{
  VERBOSE(LIR) << "* Sub" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Sub::Input::LHS).value() << ", "
               << node.getInputs().at(Sub::Input::RHS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Tanh &node)
{
  VERBOSE(LIR) << "* TanH" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Tanh::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const TopKV2 &node)
{
  VERBOSE(LIR) << "* TopKV2" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(TopKV2::Input::INPUT).value() << ")"
               << std::endl;
  VERBOSE(LIR) << "  - Outputs : Values("
               << node.getOutputs().at(TopKV2::Output::OUTPUT_VALUES).value() << ") Indices("
               << node.getOutputs().at(TopKV2::Output::OUTPUT_INDICES).value() << ")" << std::endl;
}

void OperationDumper::visit(const TransposeConv &node)
{
  std::string padding_type =
      node.param().padding.type == PaddingType::EXPLICIT ? "Explicit" : "Implicit";
  VERBOSE(LIR) << "* TransposeConv(" << padding_type << ")" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Output Shape("
               << node.getInputs().at(TransposeConv::Input::OUTPUT_SHAPE).value() << ") KERNEL("
               << node.getInputs().at(TransposeConv::Input::KERNEL).value() << ") IFM("
               << node.getInputs().at(TransposeConv::Input::INPUT).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : OFM(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Transpose &node)
{
  VERBOSE(LIR) << "* Transpose" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Transpose::Input::INPUT).value()
               << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Unpack &node)
{
  VERBOSE(LIR) << "* Unpack" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Unpack::Input::INPUT).value() << ")"
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

void OperationDumper::visit(const Min &node)
{
  VERBOSE(LIR) << "* Min" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Min::Input::LHS).value() << ", "
               << node.getInputs().at(Min::Input::RHS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const Max &node)
{
  VERBOSE(LIR) << "* Max" << std::endl;
  VERBOSE(LIR) << "  - Inputs : Input(" << node.getInputs().at(Max::Input::LHS).value() << ", "
               << node.getInputs().at(Max::Input::RHS).value() << ")" << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

void OperationDumper::visit(const OneHot &node)
{
  VERBOSE(LIR) << "* OneHot" << std::endl;
  VERBOSE(LIR) << "  - Inputs : "
               << "Indices(" << node.getInputs().at(OneHot::Input::INDICES).value() << ") "
               << std::endl;
  VERBOSE(LIR) << "  - Output : Output(" << node.getOutputs().at(0).value() << ")" << std::endl;
}

} // namespace ir
} // namespace neurun
