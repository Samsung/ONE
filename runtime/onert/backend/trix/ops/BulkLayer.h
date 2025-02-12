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

#include <exec/IFunction.h>

namespace onert::backend::trix::ops
{

class BulkLayer : public ::onert::exec::IFunction
{
public:
  BulkLayer();
  ~BulkLayer();

public:
  void configure(const std::vector<const IPortableTensor *> &inputs,
                 std::vector<IPortableTensor *> &outputs, std::string binary_path,
                 const std::shared_ptr<DevContext> &dev_context);

  void run() override;

  void prepare() override;

private:
  std::vector<const IPortableTensor *> _inputs;
  std::vector<IPortableTensor *> _outputs;

  ModelID _model_id;
  std::shared_ptr<DevContext> _dev_context;
};

} // namespace onert::backend::trix::ops

#endif // __ONERT_BACKEND_TRIX_OPS_BULKLAYER_H__
