/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRIX_OPS_BULKLAYER_H__
#define __ONERT_BACKEND_TRIX_OPS_BULKLAYER_H__

#include <backend/IPortableTensor.h>
#include "../DevContext.h"
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

class BulkLayer : public ::onert::exec::IFunction
{
public:
  BulkLayer();
  ~BulkLayer();

public:
  void configure(const std::vecotr<IPortableTensor *> &inputs,
                 std::vector<IPortableTensor *> &output, string binary_path,
                 const std::shared_ptr<DevContext> &dev_context);

  void run() override;

  void prepare() override;

private:
  const std::vector<IPortableTensor *> _inputs;
  std::vector<IPortableTensor *> _outputs;

  const uint32_t _model_id;
  const npubin_meta *_meta;
  const std::shared_ptr<DevContext> _dev_context;
};

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_OPS_BULKLAYER_H__
