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

#ifndef __ONERT_COMPILER_ACL_CL_CONSTANT_INITIALIZER_H__
#define __ONERT_COMPILER_ACL_CL_CONSTANT_INITIALIZER_H__

#include "AclConstantInitializer.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{

class ConstantInitializer : public acl_common::AclConstantInitializer
{
public:
  ConstantInitializer(const ir::Operands &operands,
                      const std::shared_ptr<ITensorRegistry> &tensor_reg);

public:
  using acl_common::AclConstantInitializer::visit;
  void visit(const ir::operation::EmbeddingLookup &) final;
  void visit(const ir::operation::Gather &) final;
  void visit(const ir::operation::HashtableLookup &) final;
  void visit(const ir::operation::SpaceToBatchND &) final;
};

} // namespace acl_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_COMPILER_ACL_CL_CONSTANT_INITIALIZER_H__
