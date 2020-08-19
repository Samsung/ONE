/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConstantInitializer.h"

namespace onert
{
namespace backend
{
namespace acl_neon
{

ConstantInitializer::ConstantInitializer(const ir::Operands &operands,
                                         const std::shared_ptr<ITensorRegistry> &tensor_reg)
    : IConstantInitializer{operands}, _tensor_reg{tensor_reg}
{
  // DO NOTHING
}

void ConstantInitializer::copyInputInitialize(const ir::Operation &node, uint32_t index)
{
  assert(node.getInputs().size() > index);

  const auto &input_index = node.getInputs().at(index);
  const auto &input_obj = _operands.at(input_index);
  registerCopyInitializer(input_index, input_obj);
}

void ConstantInitializer::permuteInputInitialize(const ir::Operation &node, uint32_t index)
{
  assert(node.getInputs().size() > index);

  const auto &input_index = node.getInputs().at(index);
  const auto &input_obj = _operands.at(input_index);
  registerPermuteInitializer(input_index, input_obj);
}

void ConstantInitializer::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto &block_size_index = node.getInputs().at(ir::operation::BatchToSpaceND::BLOCK_SIZE);
  const auto &block_size_obj = _operands.at(block_size_index);

  if (block_size_obj.isConstant())
  {
    _init_map[block_size_index] = [](const ir::Operand &model_obj, backend::ITensor &obj) {
      assert(model_obj.data());
      const auto &shape = model_obj.shape();
      const auto base = reinterpret_cast<const int32_t *>(model_obj.data()->base());
      assert(model_obj.shape().rank() == 1);
      obj.access([&](ITensor &tensor) {
        for (size_t i = 0; i < shape.num_elements(); ++i)
        {
          const int32_t value = base[shape.num_elements() - i - 1];
          int32_t *into = reinterpret_cast<int32_t *>(tensor.buffer() +
                                                      tensor.calcOffset({static_cast<int32_t>(i)}));
          *into = value;
        }
      });
    };
  }
}

void ConstantInitializer::visit(const ir::operation::Conv2D &node)
{
  permuteInputInitialize(node, ir::operation::Conv2D::KERNEL);
  copyInputInitialize(node, ir::operation::Conv2D::BIAS);
}

void ConstantInitializer::visit(const ir::operation::DepthwiseConv2D &node)
{
  permuteInputInitialize(node, ir::operation::DepthwiseConv2D::KERNEL);
  copyInputInitialize(node, ir::operation::DepthwiseConv2D::BIAS);
}

void ConstantInitializer::visit(const ir::operation::FullyConnected &node)
{
  copyInputInitialize(node, ir::operation::FullyConnected::WEIGHT);
  copyInputInitialize(node, ir::operation::FullyConnected::BIAS);
}

void ConstantInitializer::visit(const ir::operation::LSTM &node)
{
  copyInputInitialize(node, ir::operation::LSTM::INPUT_TO_INPUT_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::INPUT_TO_FORGET_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::INPUT_TO_CELL_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::INPUT_TO_OUTPUT_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::RECURRENT_TO_INPUT_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::RECURRENT_TO_FORGET_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::RECURRENT_TO_CELL_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::RECURRENT_TO_OUTPUT_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::CELL_TO_INPUT_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::CELL_TO_FORGET_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::CELL_TO_OUTPUT_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::INPUT_GATE_BIAS);
  copyInputInitialize(node, ir::operation::LSTM::FORGET_GATE_BIAS);
  copyInputInitialize(node, ir::operation::LSTM::OUTPUT_GATE_BIAS);
  copyInputInitialize(node, ir::operation::LSTM::PROJECTION_WEIGHTS);
  copyInputInitialize(node, ir::operation::LSTM::PROJECTION_BIAS);
}

void ConstantInitializer::visit(const ir::operation::RNN &node)
{
  copyInputInitialize(node, ir::operation::RNN::WEIGHTS);
  copyInputInitialize(node, ir::operation::RNN::RECURRENT_WEIGHTS);
  copyInputInitialize(node, ir::operation::RNN::BIAS);
}

void ConstantInitializer::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto &block_size_index = node.getInputs().at(ir::operation::SpaceToBatchND::BLOCK_SIZE);
  const auto &block_size_obj = _operands.at(block_size_index);

  if (block_size_obj.isConstant())
  {
    _init_map[block_size_index] = [](const ir::Operand &model_obj, backend::ITensor &obj) {
      assert(model_obj.data());
      const auto &shape = model_obj.shape();
      const auto base = reinterpret_cast<const int32_t *>(model_obj.data()->base());
      assert(model_obj.shape().rank() == 1);
      obj.access([&](ITensor &tensor) {
        for (size_t i = 0; i < shape.num_elements(); ++i)
        {
          const int32_t value = base[shape.num_elements() - i - 1];
          int32_t *into = reinterpret_cast<int32_t *>(tensor.buffer() +
                                                      tensor.calcOffset({static_cast<int32_t>(i)}));
          *into = value;
        }
      });
    };
  }

  const auto &paddings_index = node.getInputs().at(ir::operation::SpaceToBatchND::PADDINGS);
  const auto &paddings_obj = _operands.at(paddings_index);
  if (paddings_obj.isConstant())
  {
    _init_map[paddings_index] = [](const ir::Operand &model_obj, backend::ITensor &obj) {
      assert(model_obj.data());
      const auto &shape = model_obj.shape();
      const auto base = reinterpret_cast<const int32_t *>(model_obj.data()->base());
      assert(model_obj.shape().rank() == 2);
      assert(shape.dim(0) == 2);
      assert(shape.dim(1) == 2);
      obj.access([&](ITensor &tensor) {
        for (auto i = 0; i < shape.dim(0); ++i)
        {
          for (auto j = 0; j < shape.dim(1); ++j)
          {
            const int32_t value = base[i * 2 + j];
            int32_t *into = reinterpret_cast<int32_t *>(
                // The coordinates of NETensor are different from the coordiantes of CLTensor in
                // this operand.
                // NEON : {j, reversed i}
                // CL : {reversed i, j}
                tensor.buffer() + tensor.calcOffset({j, shape.dim(0) - i - 1}));
            *into = value;
          }
        }
      });
    };
  }
}

void ConstantInitializer::visit(const ir::operation::TransposeConv &node)
{
  permuteInputInitialize(node, ir::operation::TransposeConv::KERNEL);
}

} // namespace acl_neon
} // namespace backend
} // namespace onert
