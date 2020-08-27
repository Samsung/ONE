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
    : acl_common::AclConstantInitializer{operands, tensor_reg}
{
  // DO NOTHING
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

} // namespace acl_neon
} // namespace backend
} // namespace onert
