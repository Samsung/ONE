/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_RUY_CONSTANT_INITIALIZER_H__
#define __ONERT_BACKEND_RUY_CONSTANT_INITIALIZER_H__

#include "backend/cpu_common/TensorRegistry.h"

#include <backend/IConstantInitializer.h>
#include <ir/Operands.h>

namespace onert
{
namespace backend
{
namespace ruy
{

class ConstantInitializer : public IConstantInitializer
{
public:
  ConstantInitializer(const ir::Operands &operands,
                      const std::shared_ptr<ITensorRegistry> &tensor_reg);

public:
  void registerDefaultInitializer(const ir::OperandIndex &index, const ir::Operand &obj) override;

  // TODO: For now the only cpu backend supports constant tensor to use data from external
  // If the other backend supports (to do this,
  // ExternalTensor should be abstract such as IExternal, maybe),
  // this can be an interface of IConstantInitializer
  void registerExternalInitializer(const ir::OperandIndex &, const ir::Operand &);

public:
  void visit(const ir::operation::Conv2D &) override;

private:
  std::shared_ptr<ITensorRegistry> tensor_registry() const override { return _tensor_reg; }

private:
  std::shared_ptr<ITensorRegistry> _tensor_reg;
};

} // namespace ruy
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_RUY_CONSTANT_INITIALIZER_H__
