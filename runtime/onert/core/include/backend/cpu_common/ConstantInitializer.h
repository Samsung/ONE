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

#ifndef __ONERT_BACKEND_CPU_COMMON_CONSTANT_INITIALIZER_H__
#define __ONERT_BACKEND_CPU_COMMON_CONSTANT_INITIALIZER_H__

#include <unordered_map>
#include <functional>

#include "TensorRegistry.h"
#include "ir/Coordinates.h"
#include "ir/Layout.h"
#include "ir/Operand.h"
#include "ir/Operands.h"
#include "ir/OperationVisitor.h"
#include "ir/OpSequence.h"
#include "backend/ITensor.h"
#include "backend/ITensorRegistry.h"
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

class ConstantInitializer : public ir::OperationVisitor
{
public:
  ConstantInitializer(const ir::Operands &operands,
                      const std::shared_ptr<ITensorRegistry> &tensor_reg);

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
      VERBOSE(FillOperandData) << "Fill data for " << ind << std::endl;
    }
    _init_map.clear();
  }

public:
  ConstantInitializer(const ir::Operands &operands)
    : _operands{operands}, _current_layout{ir::Layout::UNKNOWN}
  {
  }

public:
  using Initializer = std::function<void(const ir::Operand &, backend::ITensor &)>;

public:
  void registerDefaultInitializer(const ir::OperandIndex &index, const ir::Operand &obj);
  void registerExternalInitializer(const ir::OperandIndex &, const ir::Operand &);
  void registerCopyInitializer(const ir::OperandIndex &index, const ir::Operand &obj);
  void registerPermuteInitializer(const ir::OperandIndex &index, const ir::Operand &obj);

public:
  void setLayout(ir::Layout layout) { _current_layout = layout; }
  bool exist(const ir::OperandIndex &ind) { return _init_map.find(ind) != _init_map.end(); }

private:
  const ir::Operands &_operands;
  std::shared_ptr<ITensorRegistry> _tensor_reg;
  std::unordered_map<ir::OperandIndex, Initializer> _init_map;
  ir::Layout _current_layout;
};

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_COMMON_CONSTANT_INITIALIZER_H__
