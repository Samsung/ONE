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

#ifndef __ONERT_BACKEND_CPU_SHAPE_FIXER_H__
#define __ONERT_BACKEND_CPU_SHAPE_FIXER_H__

#include "TensorBuilder.h"
#include "Tensor.h"

#include <backend/IShapeFixer.h>
#include <ir/Operands.h>

namespace onert
{
namespace backend
{
namespace cpu
{

class ShapeFixer : public IShapeFixer
{
public:
  ShapeFixer(const ir::Operands &ctx);

  void visit(const ir::operation::Add &) override;
  void visit(const ir::operation::Div &) override;
  void visit(const ir::operation::Pad &) override;

private:
  const ir::Operands &_ctx;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_SHAPE_FIXER_H__
