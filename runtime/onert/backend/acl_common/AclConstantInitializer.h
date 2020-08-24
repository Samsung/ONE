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

#ifndef __ONERT_COMPILER_ACL_COMMON_ACLCONSTANT_INITIALIZER_H__
#define __ONERT_COMPILER_ACL_COMMON_ACLCONSTANT_INITIALIZER_H__

#include <backend/IConstantInitializer.h>
#include <ir/Operands.h>
#include "AclTensorRegistry.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

class AclConstantInitializer : public IConstantInitializer
{
public:
  AclConstantInitializer(const ir::Operands &operands,
                         const std::shared_ptr<ITensorRegistry> &tensor_reg);

public:
  void visit(const ir::operation::BatchToSpaceND &) override;
  void visit(const ir::operation::Conv2D &) override;
  void visit(const ir::operation::DepthwiseConv2D &) override;
  void visit(const ir::operation::FullyConnected &) override;
  void visit(const ir::operation::LSTM &) override;
  void visit(const ir::operation::RNN &) override;
  void visit(const ir::operation::TransposeConv &) override;

protected:
  void copyInputInitialize(const ir::Operation &node, uint32_t index);
  void permuteInputInitialize(const ir::Operation &node, uint32_t index);

private:
  std::shared_ptr<ITensorRegistry> tensor_registry() const final { return _tensor_reg; }

protected:
  std::shared_ptr<ITensorRegistry> _tensor_reg;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_COMPILER_ACL_COMMON_ACLCONSTANT_INITIALIZER_H__
