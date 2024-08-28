/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BiasInsertionPass.h"

#include "ir/Graph.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace train
{
namespace pass
{

void BiasInsertionPass::run()
{
  _graph.operations().iterate([&](const ir::OperationIndex &op_index, const ir::IOperation &node) {
    _current_op_index = op_index;
    node.accept(*this);
  });
}

void BiasInsertionPass::visit(const ir::operation::Conv2D &) {}

void BiasInsertionPass::visit(const ir::operation::DepthwiseConv2D &) {}

void BiasInsertionPass::visit(const ir::operation::FullyConnected &node)
{
  const auto &bias_index = node.getInputs().at(ir::operation::Conv2D::Input::BIAS);

  // Insert bias if it is optional
  if (!bias_index.valid())
  {
    const auto &output_index = node.getOutputs().at(0);
    const auto &output = _graph.operands().at(output_index);
    const auto &output_shape = output.shape();
    const auto bias_shape = ir::Shape{output_shape.dim(output_shape.rank() - 1)};

    auto bias_typeinfo = output.typeInfo();
    if (bias_typeinfo.type() != ir::DataType::FLOAT32)
      throw std::runtime_error("BiasInsertionPass: Only FLOAT32 is supported for now");

    const auto new_bias_index = _graph.addOperand(bias_shape, output.typeInfo());

    // TODO Replace data with sparse data to reduce memory usage
    const auto bias_size = bias_shape.num_elements() * ir::sizeOfDataType(bias_typeinfo.type());
    std::vector<uint8_t> data_vec(bias_size, 0);
    auto data_obj = std::make_shared<ir::CachedData>(data_vec.data(), bias_size);
    _graph.setOperandValue(new_bias_index, std::move(data_obj));

    auto &bias = _graph.operands().at(new_bias_index);
    bias.insertUse(_current_op_index);
    bias.isConstant();

    _graph.operations()
      .at(_current_op_index)
      .replaceInput(ir::operation::Conv2D::Input::BIAS, new_bias_index);

    VERBOSE(BiasInsertionPass) << "Optional bias is inserted for training, bias index : "
                               << bias_index << std::endl;
  }
}

} // namespace pass
} // namespace train
} // namespace compiler
} // namespace onert
