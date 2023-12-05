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

#include <AclActivationBuilder.h>
#include <AclFunction.h>
#include <Convert.h>
#include <Swizzle.h>

#include "ConstantInitializer.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{

ConstantInitializer::ConstantInitializer(const ir::Operands &operands,
                                         const std::shared_ptr<ITensorRegistry> &tensor_reg)
  : acl_common::AclConstantInitializer{operands, tensor_reg}
{
  // DO NOTHING
}

void ConstantInitializer::visit(const ir::operation::EmbeddingLookup &node)
{
  copyInputInitialize(node, ir::operation::EmbeddingLookup::LOOKUPS);
}

void ConstantInitializer::visit(const ir::operation::Gather &node)
{
  copyInputInitialize(node, ir::operation::Gather::INDICES);
}

void ConstantInitializer::visit(const ir::operation::HashtableLookup &node)
{
  copyInputInitialize(node, ir::operation::HashtableLookup::LOOKUPS);
  copyInputInitialize(node, ir::operation::HashtableLookup::KEYS);
}

void ConstantInitializer::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto &block_size_index = node.getInputs().at(ir::operation::SpaceToBatchND::BLOCK_SIZE);
  const auto &block_size_obj = _operands.at(block_size_index);

  if (block_size_obj.isConstant())
  {
    _init_map[block_size_index] = acl_common::initReverseOrder<int32_t>;
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
      assert(obj.getShape().dim(0) == 2);
      obj.access([&](ITensor &tensor) {
        for (auto i = 0; i < shape.dim(0); ++i)
        {
          for (auto j = 0; j < shape.dim(1); ++j)
          {
            const int32_t value = base[i * 2 + j];
            int32_t *into = reinterpret_cast<int32_t *>(
              tensor.buffer() + tensor.calcOffset({shape.dim(0) - i - 1, j}));
            *into = value;
          }
        }
      });
    };
  }
}

void ConstantInitializer::visit(const ir::operation::Reverse &node)
{
  const auto &input_index = node.getInputs().at(ir::operation::Reverse::Input::INPUT);
  const auto &input_obj = _operands.at(input_index);

  const auto &axis_index = node.getInputs().at(ir::operation::Reverse::Input::AXIS);
  const auto &axis_obj = _operands.at(axis_index);

  const auto ifm_rank = input_obj.shape().rank();

  if (axis_obj.isConstant())
  {
    _init_map[axis_index] = [ifm_rank](const ir::Operand &operand, backend::ITensor &obj) {
      assert(operand.data());

      // TODO Support other types
      auto axis_value = *(reinterpret_cast<const int32_t *>(operand.data()->base()));
      if (axis_value < 0)
      {
        axis_value = axis_value + ifm_rank;
      }

      obj.access([&](ITensor &tensor) {
        int32_t *into = reinterpret_cast<int32_t *>(tensor.buffer());
        *into = axis_value;
      });
    };
  }
}

} // namespace acl_cl
} // namespace backend
} // namespace onert
