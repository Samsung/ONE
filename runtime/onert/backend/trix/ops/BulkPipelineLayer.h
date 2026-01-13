/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRIX_OPS_BULK_PIPELINE_LAYER_H__
#define __ONERT_BACKEND_TRIX_OPS_BULK_PIPELINE_LAYER_H__

#include <backend/IPortableTensor.h>
#include "../DevContext.h"
#include <exec/IFunction.h>
#include "BulkPipelineManager.h"

namespace onert::backend::trix::ops
{

class BulkPipelineLayer : public ::onert::exec::IFunction
{
public:
  BulkPipelineLayer();
  ~BulkPipelineLayer() override;

public:
  void configure(const std::vector<const IPortableTensor *> &inputs,
                 std::vector<IPortableTensor *> &outputs,
                 const std::vector<std::string> &binary_path);

  void run() override;

  void prepare() override;

private:
  std::vector<const IPortableTensor *> _inputs;
  std::vector<IPortableTensor *> _outputs;

  // Pipeline manager
  std::unique_ptr<BulkPipelineManager> _pipeline_manager;
};

} // namespace onert::backend::trix::ops

#endif // __ONERT_BACKEND_TRIX_OPS_BULK_PIPELINE_LAYER_H__
