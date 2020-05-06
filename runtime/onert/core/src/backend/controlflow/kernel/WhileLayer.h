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

#ifndef __ONERT_BACKEND_CONTROLFLOW_KERNEL_WHILE_LAYER_H__
#define __ONERT_BACKEND_CONTROLFLOW_KERNEL_WHILE_LAYER_H__

#include <backend/ITensor.h>
#include <exec/IPermuteFunction.h>
#include <exec/IExecutor.h>

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

class WhileLayer : public ::onert::exec::IPermuteFunction
{
public:
  WhileLayer(std::vector<std::shared_ptr<backend::ITensor>> input_tensors,
             std::vector<std::shared_ptr<backend::ITensor>> output_tensors,
             exec::IExecutor &cond_executor, exec::IExecutor &body_executor);

public:
  void configure();

  void optimize() override
  {
    // TODO Optimize
  }

  void run() override;

private:
  exec::IExecutor &_cond_executor;
  exec::IExecutor &_body_executor;
};

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_KERNEL_WHILE_LAYER_H__
