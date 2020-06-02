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

#ifndef __ONERT_BACKEND_ISHAPE_FIXER_H__
#define __ONERT_BACKEND_ISHAPE_FIXER_H__

#include <memory>
#include <functional>

#include "ir/LowerInfoMap.h"
#include "ITensorBuilder.h"
#include "ir/OperationVisitor.h"
#include "ir/OpSequence.h"
#include <memory>

namespace onert
{
namespace backend
{

class IShapeFixer : public ir::OperationVisitor
{
public:
  virtual ~IShapeFixer() = default;

protected:
  using OperationVisitor::visit;
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ISHAPE_FIXER_H__
