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

#ifndef __ONERT_BACKEND_XNNPACK_OPS_LAYER_H__
#define __ONERT_BACKEND_XNNPACK_OPS_LAYER_H__

#include <exec/IFunction.h>
#include <backend/IPortableTensor.h>
#include "OperationUtils.h"
#include "../ExternalContext.h"

#include <cassert>
#include <memory>

#include <xnnpack.h>

namespace onert
{
namespace backend
{
namespace xnnpack
{
namespace ops
{

class Layer : public ::onert::exec::IFunction
{
public:
  Layer(const std::shared_ptr<ExternalContext> external_context)
      : _kernel_op{nullptr}, _prepare{false}, _external_context{external_context}
  {
    // DO NOTHING
  }

  ~Layer()
  {
    if (_kernel_op)
      xnn_delete_operator(_kernel_op);
  }

protected:
  xnn_operator_t _kernel_op;
  bool _prepare;
  const std::shared_ptr<ExternalContext> _external_context;
};

} // namespace ops
} // namespace xnnpack
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_XNNPACK_OPS_LAYER_H__
