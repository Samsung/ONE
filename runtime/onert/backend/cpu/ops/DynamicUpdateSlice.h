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

#ifndef __ONERT_BACKEND_CPU_OPS_DYNAMIC_UPDATE_SLICE_LAYER_H__
#define __ONERT_BACKEND_CPU_OPS_DYNAMIC_UPDATE_SLICE_LAYER_H__

#include <backend/IPortableTensor.h>
#include <exec/IFunction.h>
#include <ir/operation/DynamicUpdateSlice.h>

namespace onert::backend::cpu::ops
{

class DynamicUpdateSliceLayer : public ::onert::exec::IFunction
{
public:
  DynamicUpdateSliceLayer();
  ~DynamicUpdateSliceLayer();

public:
  void configure(const IPortableTensor *operand, const IPortableTensor *update,
                 const IPortableTensor *indices, IPortableTensor *output);

  void run() override;

private:
  const IPortableTensor *_operand;
  const IPortableTensor *_update;
  const IPortableTensor *_indices;
  IPortableTensor *_output;
};

} // namespace onert::backend::cpu::ops

#endif // __ONERT_BACKEND_CPU_OPS_DYNAMIC_UPDATE_SLICE_LAYER_H__
