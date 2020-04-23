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

#ifndef __ONERT_BACKEND_CPU_KERNEL_PERMUTE_LAYER_H__
#define __ONERT_BACKEND_CPU_KERNEL_PERMUTE_LAYER_H__

#include "OperationUtils.h"

#include <exec/IPermuteFunction.h>
#include <ir/Coordinates.h>
#include <ir/operation/Permute.h>
#include <misc/feature/IndexIterator.h>
#include <util/feature/nchw/View.h>
#include <util/feature/nhwc/Reader.h>
#include <util/feature/nhwc/View.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class PermuteLayer : public ::onert::exec::IPermuteFunction
{
public:
  PermuteLayer() = default;

public:
  void configure(std::shared_ptr<backend::ITensor> input, std::shared_ptr<backend::ITensor> output,
                 size_t rank);

  void optimize() override
  {
    // DO NOTHING
    if (_src_tensors.at(0) == _dst_tensors.at(0))
    {
      _src_tensors.clear();
      _dst_tensors.clear();
      _ranks.clear();
    }
  }
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_KERNEL_PERMUTE_LAYER_H__
