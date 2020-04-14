/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_ITENSOR_BUILDER_H__
#define __ONERT_BACKEND_ITENSOR_BUILDER_H__

#include <map>

#include "ir/Index.h"
#include "ir/OperandInfo.h"
#include "ir/Operation.h"
#include "ir/Layout.h"
#include "ITensor.h"
#include "ITensorManager.h"

namespace onert
{
namespace backend
{

struct ITensorBuilder
{
  using IterateFunction = std::function<void(const ir::OperandIndex &)>;

  virtual ~ITensorBuilder(void) = default;

  /**
   * @brief Register tensor information to allocate on backend
   */
  virtual void registerTensorInfo(const ir::OperandIndex &, const ir::OperandInfo &,
                                  ir::Layout backend_layout, bool as_const) = 0;

  virtual void notifyFirstUse(const ir::OperandIndex &) = 0;
  virtual void notifyLastUse(const ir::OperandIndex &) = 0;

  virtual bool isRegistered(const ir::OperandIndex &) const = 0;

  virtual void prepare(void) = 0;
  virtual void allocate() = 0;
  virtual void postFunctionPrepare() = 0;

  virtual std::shared_ptr<ITensor> tensorAt(const ir::OperandIndex &ind) = 0;
  virtual void iterate(const IterateFunction &fn) = 0;

  virtual std::unique_ptr<ITensorManager> releaseTensorManager(void) = 0;
};

} // namespace backend
} // namespace onert

#include <unordered_set>
#include <memory>

namespace onert
{
namespace backend
{

using TensorBuilderSet = std::unordered_set<std::shared_ptr<backend::ITensorBuilder>>;

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ITENSOR_BUILDER_H__
