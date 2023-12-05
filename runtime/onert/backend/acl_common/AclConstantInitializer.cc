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

#include "AclConstantInitializer.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

AclConstantInitializer::AclConstantInitializer(const ir::Operands &operands,
                                               const std::shared_ptr<ITensorRegistry> &tensor_reg)
  : _operands{operands}, _tensor_reg{tensor_reg}
{
  // DO NOTHING
}

void AclConstantInitializer::copyInputInitialize(const ir::Operation &node, uint32_t index)
{
  assert(node.getInputs().size() > index);

  const auto &input_index = node.getInputs().at(index);
  if (input_index.valid())
  {
    const auto &input_obj = _operands.at(input_index);
    registerCopyInitializer(input_index, input_obj);
  }
}

void AclConstantInitializer::permuteInputInitialize(const ir::Operation &node, uint32_t index)
{
  assert(node.getInputs().size() > index);

  const auto &input_index = node.getInputs().at(index);
  const auto &input_obj = _operands.at(input_index);
  registerPermuteInitializer(input_index, input_obj);
}

void AclConstantInitializer::visit(const ir::operation::BatchToSpaceND &node)
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

void AclConstantInitializer::visit(const ir::operation::Conv2D &node)
{
  // OHWI -> WHIO
  permuteInputInitialize(node, ir::operation::Conv2D::KERNEL);
  copyInputInitialize(node, ir::operation::Conv2D::BIAS);
}

void AclConstantInitializer::visit(const ir::operation::DepthwiseConv2D &node)
{
  // OHWI -> WHIO
  permuteInputInitialize(node, ir::operation::DepthwiseConv2D::KERNEL);
  copyInputInitialize(node, ir::operation::DepthwiseConv2D::BIAS);
}

void AclConstantInitializer::visit(const ir::operation::FullyConnected &node)
{
  copyInputInitialize(node, ir::operation::FullyConnected::WEIGHT);
  copyInputInitialize(node, ir::operation::FullyConnected::BIAS);
}

void AclConstantInitializer::visit(const ir::operation::LSTM &node)
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

void AclConstantInitializer::visit(const ir::operation::RNN &node)
{
  copyInputInitialize(node, ir::operation::RNN::WEIGHTS);
  copyInputInitialize(node, ir::operation::RNN::RECURRENT_WEIGHTS);
  copyInputInitialize(node, ir::operation::RNN::BIAS);
}

void AclConstantInitializer::visit(const ir::operation::TransposeConv &node)
{
  // OHWI -> WHIO
  permuteInputInitialize(node, ir::operation::TransposeConv::KERNEL);
}

// NOTE Workaround for 16b float type. Here, this is enough since only the size of bytes matters.
using float16 = uint16_t;

void AclConstantInitializer::registerCopyInitializer(const ir::OperandIndex &index,
                                                     const ir::Operand &obj)
{
  // For only CONSTANTS
  // TODO Add to check if tensor has been allocated
  if (!obj.isConstant())
    return;

  const auto type = obj.typeInfo().type();
  using ir::DataType;

  switch (type)
  {
    case DataType::FLOAT32:
      _init_map[index] = copyInit<float>;
      break;
    case DataType::INT32:
      _init_map[index] = copyInit<int32_t>;
      break;
    case DataType::UINT32:
      _init_map[index] = copyInit<uint32_t>;
      break;
    case DataType::BOOL8:
    case DataType::QUANT_UINT8_ASYMM:
      _init_map[index] = copyInit<uint8_t>;
      break;
    case DataType::QUANT_INT8_SYMM:
    case DataType::QUANT_INT8_ASYMM:
      _init_map[index] = copyInit<int8_t>;
      break;
    case DataType::FLOAT16:
      _init_map[index] = copyInit<float16>;
      break;
    case DataType::INT64:
      _init_map[index] = copyInit<int64_t>;
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

void AclConstantInitializer::registerPermuteInitializer(const ir::OperandIndex &index,
                                                        const ir::Operand &obj)
{
  // For only CONSTANTS
  // TODO Add to check if tensor has been allocated
  if (!obj.isConstant())
    return;

  const auto type = obj.typeInfo().type();
  const auto frontend_layout = obj.info().layout();
  using ir::DataType;
  using namespace std::placeholders;

  switch (type)
  {
    case DataType::FLOAT32:
      _init_map[index] = std::bind(permuteInit<float>, _1, _2, frontend_layout);
      break;
    case DataType::INT32:
      _init_map[index] = std::bind(permuteInit<int32_t>, _1, _2, frontend_layout);
      break;
    case DataType::UINT32:
      _init_map[index] = std::bind(permuteInit<uint32_t>, _1, _2, frontend_layout);
      break;
    case DataType::BOOL8:
    case DataType::QUANT_UINT8_ASYMM:
      _init_map[index] = std::bind(permuteInit<uint8_t>, _1, _2, frontend_layout);
      break;
    case DataType::QUANT_INT8_SYMM:
    case DataType::QUANT_INT8_ASYMM:
      _init_map[index] = std::bind(permuteInit<int8_t>, _1, _2, frontend_layout);
      break;
    case DataType::FLOAT16:
      _init_map[index] = std::bind(permuteInit<float16>, _1, _2, frontend_layout);
      break;
    case DataType::INT64:
      _init_map[index] = std::bind(permuteInit<int64_t>, _1, _2, frontend_layout);
      break;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

} // namespace acl_common
} // namespace backend
} // namespace onert
