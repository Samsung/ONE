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

#ifndef __ONERT_BACKEND_ACL_NEON_SHAPE_FIXER_H__
#define __ONERT_BACKEND_ACL_NEON_SHAPE_FIXER_H__

#include <backend/IShapeFixer.h>

#include "ir/Operands.h"
#include "TensorBuilder.h"

namespace onert
{
namespace backend
{
namespace acl_neon
{

class ShapeFixer : public IShapeFixer
{
public:
  ShapeFixer(const ir::Operands &ctx, const std::shared_ptr<TensorBuilder> &tensor_builder);

  void visit(const ir::operation::LogicalAnd &) override;
  void visit(const ir::operation::LogicalOr &) override;
  void visit(const ir::operation::Pack &) override;
  void visit(const ir::operation::Mul &) override;
  void visit(const ir::operation::PReLU &) override;
  void visit(const ir::operation::Comparison &) override;
  void visit(const ir::operation::SquaredDifference &) override;
  void visit(const ir::operation::Sub &) override;
  void visit(const ir::operation::Add &) override;
  void visit(const ir::operation::Div &) override;
  void visit(const ir::operation::Min &) override;
  void visit(const ir::operation::Max &) override;

private:
  const ir::Operands &_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
};

} // namespace acl_neon
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_NEON_SHAPE_FIXER_H__
