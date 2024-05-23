/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_GPU_CL_CLCONSTANT_INITIALIZER_H__
#define __ONERT_COMPILER_GPU_CL_CLCONSTANT_INITIALIZER_H__

#include <unordered_map>
#include <functional>

#include <ir/Coordinates.h>
#include <ir/Layout.h>
#include <ir/Operand.h>
#include <ir/Operands.h>
#include <ir/OperationVisitor.h>
#include <backend/ITensorRegistry.h>
#include <util/logging.h>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

template <typename T>
static void Init(const onert::ir::Operand &model_obj, onert::backend::ITensor &obj, const bool copy,
                 const onert::ir::Layout frontend_layout = onert::ir::Layout::UNKNOWN)
{
  const auto &shape = model_obj.shape();
  assert(model_obj.data());
  obj.access([&](::onert::backend::ITensor &tensor) {
    switch (shape.rank())
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
        if (copy)
        {
          tensor.enqueueWriteBuffer(model_obj.data()->base(), true);
        }
        else
        {
          // NYI
          (void)frontend_layout;
          throw std::runtime_error{"Not yet supported"};
        }
        break;
      default:
        throw std::runtime_error{"Not yet supported"};
    }
  });
}

template <typename T>
void copyInit(const onert::ir::Operand &model_obj, onert::backend::ITensor &obj)
{
  Init<T>(model_obj, obj, true);
}

template <typename T>
void permuteInit(const onert::ir::Operand &model_obj, onert::backend::ITensor &obj,
                 const onert::ir::Layout frontend_layout)
{
  const bool copy = frontend_layout == obj.layout();
  Init<T>(model_obj, obj, copy, frontend_layout);
}

class ClConstantInitializer : public ir::OperationVisitor
{
public:
  void run()
  {
    assert(_tensor_reg);
    for (const auto &it : _init_map)
    {
      const auto &ind = it.first;
      const auto &fn = it.second;

      const auto &model_obj = _operands.at(ind);
      auto tensor_obj = _tensor_reg->getNativeITensor(ind);
      assert(tensor_obj != nullptr);
      fn(model_obj, *tensor_obj);
      VERBOSE(FillOperandData) << "Fill data for operand " << ind << std::endl;
    }
    _init_map.clear();
  }

public:
  ClConstantInitializer(const ir::Operands &operands,
                        const std::shared_ptr<ITensorRegistry> &tensor_reg);

public:
  using Initializer = std::function<void(const ir::Operand &, backend::ITensor &)>;

public:
  void registerDefaultInitializer(const ir::OperandIndex &index, const ir::Operand &obj)
  {
    registerPermuteInitializer(index, obj);
  }
  void registerCopyInitializer(const ir::OperandIndex &index, const ir::Operand &obj);
  void registerPermuteInitializer(const ir::OperandIndex &index, const ir::Operand &obj);

public:
  void setLayout(ir::Layout layout) { _current_layout = layout; }
  bool exist(const ir::OperandIndex &ind) { return _init_map.find(ind) != _init_map.end(); }

public:
protected:
  void copyInputInitialize(const ir::Operation &node, uint32_t index);
  void permuteInputInitialize(const ir::Operation &node, uint32_t index);

protected:
  const ir::Operands &_operands;
  std::shared_ptr<ITensorRegistry> _tensor_reg;
  std::unordered_map<ir::OperandIndex, Initializer> _init_map;
  ir::Layout _current_layout;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_COMPILER_GPU_CL_CLCONSTANT_INITIALIZER_H__
